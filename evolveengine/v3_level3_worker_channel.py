"""Host-to-host Level 3 Worker framing for an already authenticated transport.

This is not an Agent or a listening service. The caller authenticates the
worker host out-of-band and owns transport timeouts, cancellation and closure.
The worker never accepts a command, path, arbitrary build artifact, or retry.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from struct import Struct
from threading import Event, Lock

from .v3_harness import canonical, digest
from .v3_level3_run import OPS, WorkerResult, _digest
from .v3_level3_workload import EXTENDED_PAIRS, MAX_PAIRS

SCHEMA = 'level3-native-worker-channel-v1'
HEADER = Struct('>I')
REQUEST = 64 << 20
REPLY = 128 << 10


class WorkerChannelError(RuntimeError):
    pass


def need(ok, reason):
    if not ok:
        raise WorkerChannelError(reason)


def _read(stream, size):
    result = bytearray()
    while len(result) != size:
        remaining = size-len(result)
        chunk = stream.read(min(remaining,1<<20))
        need(type(chunk) is bytes and 0 < len(chunk) <= remaining,'worker channel closed or invalid frame')
        result.extend(chunk)
    return result


def _get(stream, cap):
    size, = HEADER.unpack(_read(stream,HEADER.size))
    need(0 < size <= cap,'oversized worker channel frame')
    raw = _read(stream,size)
    try:
        payload = json.loads(raw)
        need(type(payload) is dict and canonical(payload)==raw,'noncanonical worker channel envelope')
    except (ValueError,TypeError,RecursionError) as exc:
        raise WorkerChannelError('invalid canonical worker channel envelope') from exc
    return payload


def _send(stream, payload, cap):
    raw = canonical(payload)
    need(0 < len(raw) <= cap,'oversized worker channel frame')
    for block in (HEADER.pack(len(raw)),raw):
        view = memoryview(block)
        while view:
            n = stream.write(view)
            need(type(n) is int and 0 < n <= len(view),'short Worker channel write')
            view = view[n:]
    stream.flush()


def _binding(contract, registration):
    need(_digest(registration),'missing trusted Worker registration fingerprint')
    return dict(schema=SCHEMA,registration=registration,e0=contract.from_args({'files':{}}).digest)


def _check_result(result, model, operation, build_id, cap):
    need(type(result) is WorkerResult and (result.engine,result.operation,result.build)==(model,operation,build_id)
         and _digest(result.evidence) and type(result.data) is dict and type(result.node_ms) is int
         and 0 <= result.node_ms <= cap and type(result.status) is str and 0 < len(result.status) <= 64,
         'worker returned an unbound result')


@dataclass(frozen=True)
class WorkerLease:
    node_ms: int | None
    jobs: int | None
    max_per_op: dict

    def __post_init__(self):
        metered = self.node_ms is None and self.jobs is None
        need((metered or type(self.node_ms) is int and self.node_ms > 0 and type(self.jobs) is int and self.jobs > 0) and
             type(self.max_per_op) is dict and 0 < len(self.max_per_op) and self.max_per_op.keys() <= OPS and
             all(type(x) is int and x > 0 and (metered or x <= self.node_ms) for x in self.max_per_op.values()),
             'invalid independent Worker authorization')


def serve(source, sink, contract, worker, registration, lease, *, e0_only=False, monitor=nullcontext):
    """Run on the trusted worker under a fixed, independently authorized lease."""
    need(type(lease) is WorkerLease and type(e0_only) is bool,'no independent Worker lease')
    limits, remaining, jobs = dict(lease.max_per_op), lease.node_ms, lease.jobs
    binding = _binding(contract,registration)
    records = {}
    seq = 1
    failed_verification = False
    _send(sink,dict(kind='hello',**binding),REPLY)
    while True:
        try:
            frame = _get(source, REQUEST)
            serial = frame.pop('seq',None)
            need(type(serial) is int and serial==seq and frame.pop('registration',None)==registration,
                 'invalid Worker channel registration or sequence')
            op = frame.pop('kind',None)
            if op=='close':
                need(not frame,'invalid close request')
                _send(sink,dict(seq=seq,kind='closed'),REPLY)
                return
            need(not failed_verification,'worker evidence already failed verification')
            if op=='artifact_query':
                need(set(frame)=={'query'} and type(frame['query']) is dict
                     and frame['query'].keys() <= {'view','identity','offset','limit'},
                     'invalid artifact query envelope')
                try:
                    if not hasattr(worker,'artifact_query'):
                        raise ValueError('development artifacts unavailable')
                    with monitor():
                        page = worker.artifact_query(**frame['query'])
                    response = dict(seq=seq,kind='artifact_page',page=page)
                except (ValueError,TypeError,OSError) as exc:
                    # Invalid Agent page input does not invalidate a scoring receipt.
                    response = dict(seq=seq,kind='artifact_page',error=type(exc).__name__)
            elif op=='verify':
                need(set(frame)=={'evidence','receipt_sha256'} and _digest(frame['evidence']) and
                     _digest(frame['receipt_sha256']),'invalid verification request')
                result = records.get(frame['evidence'])
                ok = (result is not None and digest(canonical(asdict(result)))==frame['receipt_sha256'] and
                      worker.verify(result) is True)
                response = dict(seq=seq,kind='verified',verified=ok)
                failed_verification = not ok
            else:
                source_text = pair_index = workload = None
                if op in ('run', 'independent_build'):
                    need(set(frame)=={'operation','engine','candidate','parent_build','cap_ms'},
                         'invalid Worker operation request')
                    operation = frame['operation']
                    if op == 'independent_build':
                        need(operation == 'develop_build' and frame['parent_build'] is None,
                             'independent cold verification requires a build with no parent')
                elif op=='sampling':
                    from .v3_level3_sampling import challenge
                    need(set(frame)=={'operation','engine','candidate','parent_build','cap_ms','challenge'} and frame['operation']=='develop_quality','invalid sampling envelope')
                    challenge(frame['challenge']); operation='develop_quality'
                elif op in ('measure', 'stress'):
                    need(set(frame)=={'operation','engine','candidate','parent_build','cap_ms','pair_index'} and
                         frame['operation']=='develop_measure' and type(frame['pair_index']) is int and
                         0 <= frame['pair_index'] < (EXTENDED_PAIRS if jobs is None else MAX_PAIRS),
                         'invalid Worker timing request')
                    need(op != 'stress' or jobs is None and 'develop_profile' in limits,
                         'stress operation requires objective engineering scope')
                    operation, pair_index = frame['operation'], frame['pair_index']
                elif op == 'diagnose':
                    need(set(frame)=={'operation','engine','candidate','parent_build','cap_ms','source'}
                         and frame.get('operation')=='develop_diagnostic' and type(frame.get('source')) is str
                         and 0 < len(frame['source'].encode('utf-8')) <= ((2 << 20) if jobs is None else (24 << 10)),
                         'invalid bounded diagnostic request')
                    operation,source_text = frame['operation'],frame['source']
                elif op == 'profile':
                    need(set(frame)=={'operation','engine','candidate','parent_build','cap_ms','workload'}
                         and frame['operation']=='develop_profile' and type(frame['workload']) is dict,
                         'invalid development profile request')
                    operation, workload = frame['operation'], frame['workload']
                else:
                    need(False,'unregistered Worker request kind')
                cap, build_id = frame['cap_ms'],frame['parent_build']
                need(type(operation) is str and operation in limits and type(cap) is int and
                     (jobs is None or jobs > 0) and 0 < cap <= limits[operation] and (remaining is None or cap <= remaining),'Worker authorization exceeded')
                need((operation == 'develop_diagnostic') == (op == 'diagnose'),
                     'diagnostic requires its separate bounded channel operation')
                need((operation == 'develop_profile') == (op == 'profile'),
                     'profile requires its authorized non-scoring operation')
                candidate = contract.from_args(frame['candidate'])
                need(not e0_only or candidate.digest==binding['e0'],'engineering Worker only accepts E0')
                need(candidate.digest==frame['engine'] and
                     (build_id is None and operation=='develop_build' or _digest(build_id)),
                     'invalid candidate or parent identity')
                build = None if build_id is None else records.get(build_id)
                need(build_id is None or (build is not None and (operation=='develop_build' or build.engine==candidate.digest) and
                     build.operation=='develop_build' and build.status=='built' and worker.verify(build) is True),
                     'parent does not identify a verified Worker build')
                if jobs is not None:
                    jobs -= 1
                    remaining -= cap
                with monitor():
                    if op == 'independent_build':
                        result = worker.independent_build(candidate,cap)
                        need(result.data.get('build_mode') == 'independent_cold',
                             'worker did not run an independent cold build')
                    elif op == 'diagnose':
                        result = worker.diagnose(candidate,build,source_text,cap)
                    elif op=='sampling':
                        result=worker.sampling(candidate,build,cap,frame['challenge'])
                    elif op == 'measure':
                        result = worker.measure(candidate,build,cap,pair_index)
                    elif op == 'stress':
                        result = worker.stress(candidate,build,cap,pair_index)
                    elif op == 'profile':
                        result = worker.profile(candidate,build,cap,workload)
                    else:
                        result = worker(operation,candidate,build,cap)
                    # A repeated build has no parent in the returned build receipt.
                    _check_result(result,candidate.digest,operation,None if operation=='develop_build' else build_id,cap)
                    need(worker.verify(result) is True and records.get(result.evidence,result)==result,
                         'Worker result cannot be independently revalidated')
                if remaining is not None:
                    remaining += cap-result.node_ms
                records[result.evidence] = result
                response = dict(seq=seq,kind='result',result=asdict(result))
            _send(sink,response,REPLY)
            seq += 1
        except Exception as exc:
            # Controller holds the outstanding reservation. Never turn an
            # execution exception or disconnect into a zero-cost verdict.
            try:
                _send(sink,dict(seq=seq,kind='stopped'),REPLY)
            except Exception:
                pass
            if isinstance(exc,WorkerChannelError):
                raise
            raise WorkerChannelError('Worker channel stopped during request or execution') from exc


class ChannelIO:
    """Unbuffered binary adapter for an authenticated SSH session channel/socket."""
    def __init__(self, channel):
        self.channel = channel

    def read(self, size):
        return self.channel.recv(size)

    def write(self, data):
        return self.channel.send(bytes(data))

    def flush(self):
        pass


class WorkerClient:
    """Controller Worker interface; transport errors are terminal, not retried."""
    def __init__(self, source, sink, contract, registration, *, set_timeout, interrupt_transport=None, reconnect=None):
        self._source,self._sink,self._contract = source,sink,contract
        self._registration,self._set_timeout = registration,set_timeout
        self._lock,self._seq,self._stopped,self._blocked = Lock(),1,False,False
        self._interrupted = Event()
        self._interrupt_transport = interrupt_transport
        self._reconnect = reconnect
        set_timeout(75 if reconnect else 30)
        need(_get(source,REPLY)==dict(kind='hello',**_binding(contract,registration)),
             'worker channel does not match pinned E0 and registration')

    @classmethod
    def from_channel(cls, channel, contract, registration):
        """Attach after host authentication and start of the pinned Worker entrypoint."""
        io = ChannelIO(channel)
        return cls(io, io, contract, registration, set_timeout=channel.settimeout,
                   interrupt_transport=getattr(channel, 'shutdown_write', None))

    def interrupt(self):
        """Ask the remote controller watcher to stop, even during an in-flight call."""
        # The in-flight RPC holds _lock until the Worker replies or exits.
        if not self._interrupted.is_set():
            self._interrupted.set()
            if self._interrupt_transport is not None:
                self._interrupt_transport()

    def _call(self, kind, expected, seconds, **values):
        with self._lock:
            need(not self._interrupted.is_set() and not self._stopped and (not self._blocked or kind=='close'),
                 'worker channel not open for this operation')
            seq = self._seq
            frame = dict(seq=seq,kind=kind,registration=self._registration,**values)
            try:
                for attempt in range(5 if self._reconnect else 1):
                    try:
                        if attempt:
                            need(not self._interrupted.wait(min(40,5 * 2**(attempt-1))), 'Worker interrupted during reconnect')
                            self._source,self._sink,self._set_timeout = self._reconnect()
                            self._set_timeout(75)
                            need(_get(self._source, REPLY) == dict(kind='hello',**_binding(self._contract,self._registration)),
                                 'reopened Worker has a different identity')
                        self._set_timeout(seconds)
                        _send(self._sink, frame, REQUEST)
                        row = _get(self._source, REPLY)
                        break
                    except (OSError, WorkerChannelError, EOFError):
                        if self._interrupted.is_set() or attempt == (4 if self._reconnect else 0):
                            raise
                need(not self._interrupted.is_set(), 'worker reply arrived after interruption')
                serial = row.pop('seq',None)
                need(type(serial) is int and serial==seq and row.pop('kind',None)==expected,
                     'worker stopped or response identity differs')
            except Exception as exc:
                self._stopped = True
                raise WorkerChannelError('worker channel failed; do not redispatch') from exc
            self._seq += 1
            self._stopped = kind=='close'
            if kind=='verify' and row!={'verified':True}:
                self._blocked = True
            return row

    def __call__(self, operation, engine, build, cap):
        return self._run('run', operation, engine, build, cap)

    def independent_build(self, engine, cap):
        return self._run('independent_build', 'develop_build', engine, None, cap)

    def sampling(self,engine,build,cap,challenge):
        return self._run('sampling','develop_quality',engine,build,cap,challenge=challenge)

    def _run(self, kind, operation, engine, build, cap, **extra):
        parent = None if build is None else build.evidence
        row = self._call(kind,'result',max(30,cap/1000+15),operation=operation,engine=engine.digest,
                         candidate=self._contract.to_args(engine),parent_build=parent,cap_ms=cap,**extra)
        try:
            need(set(row)=={'result'},'invalid Worker result envelope')
            result = WorkerResult(**row['result'])
            _check_result(result,engine.digest,operation,None if operation=='develop_build' else parent,cap)
            if kind == 'independent_build':
                need(result.data.get('build_mode') == 'independent_cold','unverified cold build mode')
            return result
        except Exception:
            self._stopped = True
            raise

    def measure(self, engine, build, cap, pair_index):
        return self._timing('measure', engine, build, cap, pair_index)

    def stress(self, engine, build, cap, pair_index):
        return self._timing('stress', engine, build, cap, pair_index)

    def _timing(self, kind, engine, build, cap, pair_index):
        parent = build.evidence
        row = self._call(kind,'result',max(30,cap/1000+15),operation='develop_measure',engine=engine.digest,
                         candidate=self._contract.to_args(engine),parent_build=parent,cap_ms=cap,pair_index=pair_index)
        try:
            need(set(row)=={'result'},'invalid Worker timing envelope')
            result = WorkerResult(**row['result'])
            _check_result(result,engine.digest,'develop_measure',parent,cap)
            return result
        except Exception:
            self._stopped = True
            raise

    def diagnose(self, engine, build, source, cap):
        parent = None if build is None else build.evidence
        row = self._call('diagnose','result',max(30,cap/1000+15),operation='develop_diagnostic',
                         engine=engine.digest,candidate=self._contract.to_args(engine),parent_build=parent,
                         cap_ms=cap,source=source)
        try:
            need(set(row)=={'result'},'invalid Worker diagnostic envelope')
            result = WorkerResult(**row['result'])
            _check_result(result,engine.digest,'develop_diagnostic',parent,cap)
            return result
        except Exception:
            self._stopped = True
            raise

    def profile(self, engine, build, cap, workload):
        parent = build.evidence
        row = self._call('profile','result',max(30,cap/1000+15),operation='develop_profile',engine=engine.digest,
                         candidate=self._contract.to_args(engine),parent_build=parent,cap_ms=cap,workload=workload)
        need(set(row)=={'result'}, 'invalid Worker profile response')
        result = WorkerResult(**row['result'])
        _check_result(result,engine.digest,'develop_profile',parent,cap)
        return result

    def artifact_query(self, view, identity=None, offset=0, limit=None):
        query = dict(view=view,identity=identity,offset=offset,limit=limit)
        row = self._call('artifact_query','artifact_page',300,query=query)
        if 'error' in row:
            raise ValueError('development artifact query was refused')
        need(set(row)=={'page'} and type(row['page']) is dict, 'invalid development artifact response')
        return row['page']

    def verify(self, result):
        row = self._call('verify','verified',120,evidence=result.evidence,receipt_sha256=digest(canonical(asdict(result))))
        return row=={'verified':True}

    def close(self):
        need(self._call('close','closed',15)=={},'invalid Worker close acknowledgement')
