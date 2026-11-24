"""Durable host-side state transitions for a Level 3 exploration.

Only trusted outer adapters may record quality/measure outcomes and verify typed
artifact references. Nothing in this module launches a segment, tool, or evaluator.
The journal and referenced stores must never be mounted into the Pi guest.
"""
from dataclasses import asdict, dataclass
import re

from .journal import Journal

SCHEMA = 'spark26-level3-lifecycle-v1'
ROOT = 'level3/'
HEX = re.compile(r'[0-9a-f]{64}\Z')
RECOVERABLE = frozenset({'generation_truncated', 'context_generation_window',
                         'network_failure', 'guest_failure', 'worker_failure'})


def _need(ok, why):
    if not ok:
        raise ValueError(why)


def _ref(value):
    _need(type(value) is str and HEX.fullmatch(value), 'invalid artifact digest')
    return value


def _refs(values):
    _need(type(values) in (list, tuple), 'expected ordered artifact references')
    validated = tuple(_ref(item) for item in values)
    _need(len(validated) == len(set(validated)), 'duplicate artifact references')
    return validated


@dataclass(frozen=True)
class QualifiedEngine:
    engine: str
    build: str
    quality: str

    def __post_init__(self):
        for value in (self.engine, self.build, self.quality):
            _ref(value)


@dataclass(frozen=True)
class Checkpoint:
    # The harness selected for the next segment is not necessarily now loaded.
    engine: QualifiedEngine
    harness: str
    trusted_facts: str
    harness_memory: tuple[str, ...] = ()
    research: tuple[str, ...] = ()
    wip: str | None = None

    def __post_init__(self):
        _need(type(self.engine) is QualifiedEngine, 'missing qualified Engine')
        for value in (self.harness, self.trusted_facts):
            _ref(value)
        object.__setattr__(self, 'harness_memory', _refs(self.harness_memory))
        object.__setattr__(self, 'research', _refs(self.research))
        if self.wip is not None:
            _ref(self.wip)

    @classmethod
    def decode(cls, value):
        return cls(**dict(value, engine=QualifiedEngine(**value['engine'])))


class Lifecycle:
    """A single trusted controller serializes transitions using the run's journal.

    `verify_artifact(kind, digest)` returns True only after verifying bytes in the
    corresponding host-owned store. `verify_quality` binds a passing trusted receipt
    to the engine and build. `verify_measure` checks the same binding for development
    measurements (never independent Q/T). `verify_wip` additionally binds the draft
    to its base engine. These verifiers are private infrastructure, not guest RPCs.
    """

    def __init__(self, journal: Journal, *, verify_artifact, verify_quality, verify_measure,
                 verify_wip, max_recoveries: int):
        _need(type(max_recoveries) is int and max_recoveries >= 0, 'invalid recovery cap')
        self.journal = journal
        self.verify_artifact = verify_artifact
        self.verify_quality = verify_quality
        self.verify_measure = verify_measure
        self.verify_wip = verify_wip
        journal.event(ROOT + 'config', dict(schema=SCHEMA, max_recoveries=max_recoveries))
        self.max_recoveries = max_recoveries

    def record_quality_pass(self, engine: QualifiedEngine) -> None:
        """Trusted worker result only. Performance is deliberately not consulted."""
        _need(type(engine) is QualifiedEngine, 'need bound passing quality receipt')
        self._verify_engine(engine)
        self.journal.event(self._quality_key(engine), asdict(engine))

    def record_measure(self, engine: QualifiedEngine, receipt: str) -> None:
        self._qualified(engine)
        _need(self.verify_measure(engine, _ref(receipt)) is True, 'invalid development measure receipt')
        self.journal.event(ROOT + 'measure/' + receipt, dict(**asdict(engine), receipt=receipt))

    @staticmethod
    def _quality_key(engine: QualifiedEngine) -> str:
        return ROOT + 'quality/' + engine.quality

    def _verify_engine(self, engine: QualifiedEngine) -> None:
        for kind in ('engine', 'build', 'quality'):
            self._verify(kind, getattr(engine, kind))
        _need(self.verify_quality(engine) is True, 'quality did not confirm this engine and build')

    def _qualified(self, engine: QualifiedEngine) -> None:
        _need(self.journal.get_event(self._quality_key(engine)) == asdict(engine),
              'Engine/build/quality tuple not recorded as qualified')
        self._verify_engine(engine)

    def _verify(self, kind, reference):
        _need(self.verify_artifact(kind, _ref(reference)) is True, f'invalid or missing {kind} artifact')

    def _check(self, state: Checkpoint):
        self._qualified(state.engine)
        self._verify('harness', state.harness)
        self._verify('trusted_facts', state.trusted_facts)
        for item in state.harness_memory:
            self._verify('harness_memory', item)
        for item in state.research:
            self._verify('research', item)
        if state.wip is not None:
            self._verify('wip', state.wip)
            _need(self.verify_wip(state.wip, state.engine.engine) is True, 'WIP base engine mismatch')

    def save(self, state: Checkpoint) -> str:
        """Snapshot registered references only, without ending a segment."""
        self._check(state)
        return self.journal.put('level3_checkpoint', asdict(state))

    def load(self, identity: str) -> Checkpoint:
        state = Checkpoint.decode(self.journal.get('level3_checkpoint', _ref(identity)))
        self._check(state)
        return state

    def start(self, identity: str) -> int:
        self.load(identity)
        self.journal.event(ROOT + 'segment/1/start', dict(checkpoint=identity, kind='initial'))
        return 1

    def _open(self, segment):
        _need(type(segment) is int and segment > 0, 'invalid segment')
        key = ROOT + f'segment/{segment}/'
        _need(self.journal.get_event(key + 'start') is not None, 'unknown segment')
        _need(self.journal.get_event(key + 'end') is None, 'segment already ended')
        return key

    def handoff(self, segment: int, identity: str, *, active_harness: str,
                abnormal_cause: str | None = None) -> int:
        """After process exit and resource settlement, reserve exactly one successor.

        Runner must not dispatch a new Pi until previous guest cleanup and settlement
        have been checked. A stale pending successor can be revalidated via entry().
        """
        with self.journal.transaction():
            key = self._open(segment)
            state = self.load(identity)
            _need(state.harness == _ref(active_harness), 'checkpoint Harness is no longer active')
            kind = 'ordinary'
            if abnormal_cause is not None:
                _need(abnormal_cause in RECOVERABLE, 'failure is not recoverable')
                attempts = sum(row.get('kind') == 'recovery' and row['id'].endswith('/start')
                               for row in self.journal.events(ROOT + 'segment/'))
                _need(attempts < self.max_recoveries, 'recovery cap reached')
                kind = 'recovery'
            successor = segment + 1
            continuation = dict(checkpoint=identity, kind=kind, cause=abnormal_cause)
            self.journal.event(key + 'end', dict(continuation, next_segment=successor))
            self.journal.event(ROOT + f'segment/{successor}/start', continuation)
            return successor

    def entry(self, segment: int) -> Checkpoint:
        key = self._open(segment)
        return self.load(self.journal.get_event(key + 'start')['checkpoint'])

    def finish(self, segment: int, engine: QualifiedEngine, *, active_harness: str) -> dict:
        """Seal the Agent's explicit choice; never select from timings or include WIP."""
        with self.journal.transaction():
            key = self._open(segment)
            self._qualified(engine)
            self._verify('harness', active_harness)
            final = dict(kind='delivered', chosen=asdict(engine), active_harness=active_harness,
                         segment=segment)
            self.journal.event(ROOT + 'delivery', final)
            self.journal.event(key + 'end', final)
            return final

    def stop(self, segment: int, *, cause: str) -> None:
        """Close without a delivery. A stop records no fabricated performance value."""
        _need(type(cause) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', cause), 'invalid stop cause')
        with self.journal.transaction():
            key = self._open(segment)
            self.journal.event(key + 'end', dict(kind='stopped', cause=cause))
