"""Durable versions, append-only events and idempotent budget settlement."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time

from .schema import Cost, Result, canonical, digest


class InsufficientBudget(Exception):
    pass


class Interrupted(Exception):
    """Raised by a fault injection after a reservation is durable."""


class Journal:
    def __init__(self, path, on_reserved=None, *, enforce_wall=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.on_reserved = on_reserved
        self.enforce_wall = enforce_wall
        self.db = None

    def __enter__(self):
        # Separate writer lease so the journal can commit during external calls.
        self.lock = sqlite3.connect(str(self.path) + ".lock", timeout=0)
        try:
            self.lock.execute("BEGIN EXCLUSIVE")
        except sqlite3.Error:
            self.lock.close()
            raise
        try:
            self.db = sqlite3.connect(self.path, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = FULL;
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    ordinal INTEGER PRIMARY KEY,
                    id TEXT NOT NULL UNIQUE, content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, node TEXT NOT NULL,
                    input_hash TEXT NOT NULL, reserved TEXT NOT NULL,
                    started_ns INTEGER NOT NULL, physical INTEGER NOT NULL,
                    settled_ns INTEGER, receipt TEXT
                );
            """)
            self.recover()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
        self.lock.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def put(self, kind, content):
        address = digest({"kind": kind, "content": content})
        self.db.execute("INSERT OR IGNORE INTO objects VALUES (?, ?, ?)",
                        (address, kind, canonical(content)))
        return address

    def get(self, kind, address):
        row = self.db.execute("SELECT content FROM objects WHERE id=? AND kind=?",
                              (address, kind)).fetchone()
        if row is None:
            raise ValueError(f"Unknown {kind} digest: {address}")
        content = json.loads(row["content"])
        if digest({"kind": kind, "content": content}) != address:
            raise ValueError(f"Corrupt {kind} object: {address}")
        return content

    def get_event(self, key):
        row = self.db.execute("SELECT content FROM events WHERE id=?", (key,)).fetchone()
        return json.loads(row["content"]) if row else None

    def event(self, key, value):
        self.db.execute("INSERT OR IGNORE INTO events (id,content) VALUES (?,?)",
                        (key, canonical(value)))
        if canonical(self.get_event(key)) != canonical(value):
            raise ValueError(f"Cannot rewrite an event: {key}")

    def events(self, prefix):
        return [
            {"id": row["id"], **json.loads(row["content"])}
            for row in self.db.execute(
                "SELECT id,content FROM events WHERE substr(id,1,?)=? ORDER BY ordinal",
                (len(prefix), prefix),
            )
        ]

    def objects(self, kind):
        return {row["id"]: self.get(kind, row["id"])
                for row in self.db.execute("SELECT id FROM objects WHERE kind=? ORDER BY id", (kind,))}

    def budget_rows(self, prefix=""):
        for row in self.db.execute(
                "SELECT * FROM operations WHERE substr(id,1,?)=? ORDER BY started_ns,id",
                (len(prefix), prefix)):
            value = dict(row)
            value["reserved"] = json.loads(row["reserved"])
            value["receipt"] = json.loads(row["receipt"]) if row["receipt"] else None
            yield value

    def spent(self, prefix):
        total = Cost()
        for row in self.budget_rows(prefix.rstrip("/") + "/"):
            amounts = row["receipt"]["cost"] if row["receipt"] else row["reserved"]
            total += Cost(**amounts)
        return total

    def can_reserve(self, scopes, cost):
        for prefix, limit in scopes:
            spent = self.spent(prefix)
            if not (spent + cost).fits(limit):
                return False
            if self.wall_elapsed(prefix, spent) + cost.wall_ms > limit.wall_ms:
                return False
        return True

    def begin_scope(self, prefix):
        key = prefix + "/wall_start"
        if self.enforce_wall and self.get_event(key) is None:
            self.event(key, {"started_ns": time.time_ns()})

    def wall_elapsed(self, prefix, spent):
        if not self.enforce_wall:
            return spent.wall_ms
        started = self.get_event(prefix + "/wall_start")
        elapsed = (time.time_ns() - started["started_ns"]) // 10**6 if started else 0
        return max(spent.wall_ms, elapsed)

    def can_finish(self, scopes, operations):
        missing = Cost()
        for key, cap in operations:
            if self.db.execute("SELECT 1 FROM operations WHERE id=?", (key,)).fetchone() is None:
                missing += cap
        return not any(missing.values()) or self.can_reserve(scopes, missing)

    def _settle(self, key, result):
        rows = self.db.execute(
            "UPDATE operations SET receipt=?, settled_ns=? WHERE id=? AND receipt IS NULL",
            (canonical(asdict(result)), time.time_ns(), key),
        ).rowcount
        if rows != 1:
            raise ValueError(f"Operation not unsettled: {key}")

    def recover(self):
        with self.transaction():
            for row in list(self.budget_rows()):
                if row["receipt"] is None:
                    self._settle(row["id"], Result({}, Cost(**row["reserved"]), "uncertain",
                                                   "interrupted after reservation"))

    def call(self, key, request, cap, scopes, owner, node, execute, physical=True):
        request_hash = digest(request)
        with self.transaction():
            row = self.db.execute("SELECT * FROM operations WHERE id=?", (key,)).fetchone()
            if row is not None:
                if (row["input_hash"] != request_hash or row["reserved"] != canonical(cap.record())
                        or row["owner"] != owner or row["node"] != node or row["physical"] != physical):
                    raise ValueError(f"Operation ID reused with different inputs: {key}")
                if row["receipt"] is None:
                    raise ValueError(f"Operation remains unsettled: {key}")
                receipt = json.loads(row["receipt"])
                return Result(receipt["data"], Cost(**receipt["cost"]), receipt["status"], receipt["detail"])
            if not self.can_reserve(scopes, cap):
                raise InsufficientBudget(key)
            self.db.execute(
                "INSERT INTO operations VALUES (?,?,?,?,?,?,?,NULL,NULL)",
                (key, owner, node, request_hash, canonical(cap.record()), time.time_ns(), int(physical)),
            )
        if self.on_reserved:
            self.on_reserved(key)
        result = execute()
        if result.status not in {"ok", "failed", "timeout", "uncertain"}:
            raise ValueError("Invalid backend operation status")
        if not result.cost.fits(cap):
            result = Result(result.data, result.cost, "failed", "operation exceeded reservation")
        self._settle(key, result)
        return result
