"""Command-line entry points for EvolveEngine."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from .experiment import MOCK_PROTOCOL, deploy_pair
from .journal import Interrupted, Journal
from .mock import MockBackend
from .partitions import Partitions


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def mock(args):
    root = args.output
    backend = MockBackend(args.scenario)
    salt = hashlib.sha256(f"simulation-only/{args.seed}/{args.replicate}".encode()).digest()
    partitions = Partitions(args.seed, salt)

    def interrupt(key):
        if args.interrupt_at == key:
            raise Interrupted(key)

    with Journal(root / "journal.sqlite3", interrupt) as journal:
        report = deploy_pair(
            journal, backend, partitions, MOCK_PROTOCOL,
            ("mock-spark-1", "mock-spark-2", "mock-spark-3", "mock-spark-4"),
            args.replicate, {"backend": backend.version, "scenario": args.scenario, "simulated": True},
            ablation=args.ablation,
        )
    write_json(root / "report.json", report)
    decisions = [event["status"] for event in report["events"]
                 if event["id"].endswith("/decided") and "/window-" in event["id"]]
    return {"simulated": True, "report": str((root / "report.json").resolve()),
            "harness_decisions": decisions,
            "rq1": report["sealed_evaluation"]["rq1"].get("engines", {})}


def main():
    parser = argparse.ArgumentParser(prog="python -m evolveengine")
    command = parser.add_subparsers(dest="command", required=True)
    simulate = command.add_parser("mock", help="run the complete offline paired simulation")
    simulate.add_argument("--output", type=Path, default=Path(".evolveengine/mock"))
    simulate.add_argument("--seed", type=int, default=21001)
    simulate.add_argument("--replicate", type=int, default=0)
    simulate.add_argument("--scenario", default="accept",
                          choices=("accept", "reject", "deviation", "quality_failure"))
    simulate.add_argument("--ablation", action="store_true")
    simulate.add_argument("--interrupt-at", help="fault injection at the given operation ID")
    simulate.set_defaults(handler=mock)
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except Interrupted as failure:
        print(f"Simulated interruption after reservation: {failure}", file=sys.stderr)
        return 75
    except (ValueError, OSError, sqlite3.Error) as failure:
        print(f"EvolveEngine: {failure}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
