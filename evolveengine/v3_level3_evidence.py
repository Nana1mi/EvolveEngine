"""Host-only Level 3 Judge and Nsight evidence adapters.

The caller pins its policy and receipt hashes independently of the archive.
Hashes protect provenance *after* a trusted collector observes a run; they are
not signatures or proof that an untrusted candidate ran on hardware.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import csv
import hashlib
import io
import json
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
from statistics import median
from typing import Callable, Mapping, Sequence

from .v3_harness import canonical, digest
from .v3_level3_workload import input_fingerprints


SCHEMA = 'spark26-level3-evidence-v1'
BOUND_REPLAY = 'request-bound-stream-replay-v2'
OPERATIONS = ('develop_quality', 'develop_measure', 'develop_profile')
# Ratios combine these scene-specific quantities; raw latency/throughput never disappear.
DEV_SCORE_RULE = {'W1':'decode_tokens_per_s','W2':'decode_tokens_per_s',
                  'W3':'reciprocal_ttft_per_s','W4':'wave_output_tokens_per_s'}
REPORTS = ('cuda_gpu_trace', 'cuda_gpu_kern_sum', 'cuda_api_sum')
COUNTER_UNAVAILABLE = {'not_requested', 'unsupported', 'permission_denied', 'no_samples',
                       'unsupported_export_schema'}
RESIDENCY_LIMITS = (
    'File hashes identify observed files, not every byte accessed at runtime.',
    'Loaded-file observation covers the selected mappings and observation instants only.',
    'External memory samples cannot attribute allocations or exclude transient or small copies.',
    'Isolated output probes cover the frozen cases, not arbitrary native behavior.',
)


class EvidenceError(ValueError):
    """An observation cannot be verified or is incomplete."""


@dataclass(frozen=True)
class EvidencePolicy:
    """Controller-pinned inputs; do not construct this from candidate receipts.

    st_dev/inode and loader record details stay in raw artifacts. The normalized
    loaded map is file SHA-256 by logical build artifact name. All logical maps
    must be nonempty. A memory ceiling is an externally observed bound, not a
    GPU allocator/residency proof.
    """
    engine: str
    worker_id: str
    boot_id: str
    device_uuid: str
    weights: Mapping[str, str]
    loaded_artifacts: Mapping[str, str]
    quality_rule_sha256: str
    quality_cases: tuple[str, ...]
    measure_workload_sha256: str
    profile_workload_sha256: str
    scenes: Mapping[str, tuple[tuple[int, int], ...]]
    warm_waves: int
    timed_waves: int
    memory_stages: tuple[str, ...]
    memory_limit_bytes: int | None  # None only in the observation-only engineering profile.
    profile_target: tuple[str, ...]
    nsys: str = 'nsys'
    gpu_metrics_arg: str | None = None
    counter_names: tuple[str, ...] = ()
    profile_phases: tuple[str, ...] = ()
    origin: str = 'offline_fixture'
    memory_source: str | None = None
    measure_input_hashes: Mapping[str, tuple[tuple[str, ...], ...]] | None = None
    measure_pair: int | None = None
    natural_eos: bool = False
    chat_corpus_sha256: str | None = None
    w3_timed_waves: int | None = None
    sampling_required: bool = False
    replay_protocol: str | None = None
    measure_input_lengths: Mapping | None = None


@dataclass(frozen=True)
class Check:
    status: str
    reason: str = ''
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]


def _need(ok, reason):
    if not ok:
        raise EvidenceError(reason)


def _digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _uint(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _text(value):
    return type(value) is str and bool(value.strip())


def _map(value):
    return isinstance(value, Mapping) and bool(value) and all(_text(k) and _digest(v) for k, v in value.items())


def _names(values):
    return type(values) is tuple and all(_text(x) for x in values) and len(set(values)) == len(values)


def _is_reserved_windows(component):
    if hasattr(ntpath, 'isreserved'):
        return ntpath.isreserved(component)
    return PureWindowsPath(component).is_reserved()


def _safe_file(root, name):
    _need(_text(name) and '\\' not in name and ':' not in name, 'unsafe archive path')
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    _need(not (posix.is_absolute() or windows.drive) and posix.as_posix() == name
          and all(p and p not in {'.', '..'} and not p.endswith(('.', ' ')) and not _is_reserved_windows(p)
                  for p in name.split('/')), 'unsafe archive path')
    path = root
    for part in posix.parts:
        path = path / part
        _need(not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(), 'archive links not allowed')
    _need(path.is_file() and path.resolve().is_relative_to(root), 'missing or escaped archive file: ' + name)
    return path


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return _stream_sha256(stream)


@dataclass(frozen=True)
class MappedFile:
    major: int
    minor: int
    inode: int
    sha256: str


def observe_loaded_mappings(maps_text: str, expected_paths: Mapping[str, str],
                            open_mapping: Callable[[str], MappedFile]) -> dict[str, str]:
    """Use host /proc/PID/maps to select executable mappings by exact path.

    The injected reader hashes the mapping's backing file and reports fstat
    identity. Do not reopen the maps pathname (which may reference a different
    inode or mount namespace). An unreadable map is missing evidence.
    """
    _need(isinstance(expected_paths, Mapping) and expected_paths and all(
        _text(k) and type(v) is str and v.startswith('/') and not v.endswith(' (deleted)')
        for k, v in expected_paths.items()) and len(set(expected_paths.values())) == len(expected_paths),
          'invalid expected mapped file paths')
    names = {path: name for name, path in expected_paths.items()}
    seen = {}
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or parts[5] not in names or 'x' not in parts[1]:
            continue
        address, _, _, dev, inode, path = parts
        _need(re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', address) and re.fullmatch(r'[0-9a-f]+:[0-9a-f]+', dev)
              and re.fullmatch(r'[0-9]+', inode) and int(inode) > 0, 'invalid external process map entry')
        identity = (*[int(x, 16) for x in dev.split(':')], int(inode))
        _need(path not in seen or seen[path][0] == identity, 'multiple backing files under mapped path')
        seen[path] = (identity, address)
    _need(seen.keys() == names.keys(), 'missing expected executable mappings')
    result = {}
    for path, name in names.items():
        identity, address = seen[path]
        f = open_mapping(address)
        _need(isinstance(f, MappedFile) and (f.major, f.minor, f.inode) == identity and _digest(f.sha256),
              'mapped backing file identity changed or was unreadable')
        result[name] = f.sha256
    return result


def hash_linux_mapping(proc_pid: Path, address: str) -> MappedFile:
    """Host-only backing file hash; permission failures must remain missing evidence."""
    _need(type(address) is str and re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', address), 'invalid mapping address')
    _need(hasattr(os, 'major') and hasattr(os, 'minor'), 'Linux mapping observer unavailable')
    lo,hi = (int(part,16) for part in address.split('-'))
    _need(0 <= lo < hi < 1<<64,'invalid mapping interval')
    # /proc/PID/maps may zero-pad addresses, but map_files names use %lx.
    with (proc_pid / 'map_files' / f'{lo:x}-{hi:x}').open('rb') as f:
        st = os.fstat(f.fileno())
        sha = _stream_sha256(f)
    return MappedFile(os.major(st.st_dev), os.minor(st.st_dev), st.st_ino, sha)


def parse_nvidia_smi_device_memory(stdout: str, device_uuid: str) -> int:
    """Parse uuid,memory.used with noheader,nounits; unavailable is not zero."""
    lines = [r for r in csv.reader(io.StringIO(stdout), skipinitialspace=True) if r]
    _need(_text(device_uuid) and all(len(r) == 2 for r in lines), 'malformed device memory output')
    rows = [r[1].strip() for r in lines if r[0].strip() == device_uuid]
    _need(len(rows) == 1 and re.fullmatch(r'[0-9]+', rows[0]), 'device memory unavailable or ambiguous')
    value = int(rows[0]) * (1 << 20)
    _need(_uint(value), 'device memory measurement outside range')
    return value


def parse_nvidia_smi_process_memory(stdout: str, device_uuid: str) -> dict[int,int]:
    """Read gpu_uuid,pid,used_gpu_memory for a single pinned device; N/A is missing."""
    values = {}
    _need(_text(device_uuid),'missing compute-memory device identity')
    for row in csv.reader(io.StringIO(stdout),skipinitialspace=True):
        if not row:
            continue
        _need(len(row)==3 and row[0].strip()==device_uuid and re.fullmatch(r'[1-9][0-9]{0,9}',row[1].strip())
              and re.fullmatch(r'[0-9]{1,9}',row[2].strip()),'compute-process memory unavailable or ambiguous')
        pid,used = int(row[1]),int(row[2])*(1<<20)
        _need(pid not in values and used<=256*(1<<30),'duplicate process memory or implausible size')
        values[pid] = used
    return values


def _unique(pairs):
    result = {}
    for k, v in pairs:
        _need(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def _json(raw):
    def reject(value):
        raise EvidenceError('invalid JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)


def _pinned_json(root, name, pin):
    _need(_digest(pin), 'missing independent receipt hash')
    raw = _safe_file(root, name).read_bytes()
    _need(hashlib.sha256(raw).hexdigest() == pin, 'receipt hash mismatch: ' + name)
    row = _json(raw)
    _need(type(row) is dict, 'receipt must be an object')
    return row


def _artifacts(root, manifest, required):
    _need(_map(manifest) and set(required) <= manifest.keys(), 'missing raw artifacts')
    paths = {}
    for name, sha in manifest.items():
        path = _safe_file(root, name)
        _need(file_sha256(path) == sha, 'artifact hash mismatch: ' + name)
        paths[name] = path
    return paths


def nsys_collection_plan(target_argv: Sequence[str], *, nsys='nsys', gpu_metrics_arg=None) -> tuple[Command, ...]:
    """Build commands for a fresh profile-only directory, without executing them.

    target_argv and optional metrics selector are supplied by the trusted worker;
    never pass agent-provided commands. A profile run must not wrap the timing run.
    """
    _need(not isinstance(target_argv, (str, bytes)) and target_argv and all(_text(a) and '\0' not in a for a in target_argv)
          and not target_argv[0].startswith('-') and _text(nsys), 'fixed profiler target required')
    _need(gpu_metrics_arg is None or (type(gpu_metrics_arg) is str and
          re.fullmatch(r'--gpu-metrics-devices?=(?:all|\d+(?:,\d+)*)', gpu_metrics_arg)),
          'invalid profiler GPU metrics selector')
    metrics = () if gpu_metrics_arg is None else (gpu_metrics_arg,)
    return (
        Command('version', (nsys, '--version')),
        Command('profile_help', (nsys, 'profile', '--help')),
        Command('stats_help', (nsys, 'stats', '--help-reports')),
        Command('export_help', (nsys, 'export', '--help')),
        Command('capture', (nsys, 'profile', '--trace=cuda,nvtx,osrt', '--output=trace', *metrics, *target_argv)),
        Command('export', (nsys, 'export', '--type=sqlite', '--output=trace.sqlite', 'trace.nsys-rep')),
        *(Command(report, (nsys, 'stats', '--report', report, '--format', 'csv', 'trace.nsys-rep'))
          for report in REPORTS),
    )


def _nsys_help(stdout, metrics_arg):
    version = stdout['version'].decode('utf-8-sig').strip()
    _need(re.search(r'Nsight Systems version\s+\d+\.\d+', version, re.IGNORECASE), 'unrecognized Nsight CLI version')
    reports = stdout['stats_help'].decode('utf-8-sig')
    _need(all(re.search(r'(?<!\w)' + r + r'(?!\w)', reports) for r in REPORTS), 'Nsight report capability missing')
    profile_help = stdout['profile_help'].decode('utf-8-sig')
    _need('--trace' in profile_help and 'sqlite' in stdout['export_help'].decode('utf-8-sig').lower(),
          'Nsight capture or SQLite export capability not observed')
    if metrics_arg is not None:
        option = metrics_arg.split('=', 1)[0]
        _need(re.search(r'(?<![-\w])' + re.escape(option) + r'(?![-\w])', profile_help),
              'Nsight GPU metrics flag absent from profile help')
    return version


def collect_nsys_outputs(directory: Path, plan: Sequence[Command],
                         run: Callable[[tuple[str, ...], Path], CommandResult]) -> None:
    """Archive stdout/stderr and exact argv/exit codes, stopping on command failure.

    No subprocess by default: the caller owns worker isolation, lease/timeouts,
    and byte limits. The directory must already exist, be empty and host-owned.
    A command failure or runner exception leaves partial artifacts for diagnosis.
    """
    _need(directory.is_dir() and not any(directory.iterdir()), 'fresh profiler directory required')
    _need(all(re.fullmatch(r'[a-z][a-z0-9_]*', cmd.name) for cmd in plan)
          and len({cmd.name for cmd in plan}) == len(plan) and bool(plan), 'invalid profiler plan')
    help_outputs = {}
    with (directory / 'commands.jsonl').open('xb') as log:
        for cmd in plan:
            if cmd.name == 'capture':
                metrics_arg = next((a for a in cmd.argv if a.startswith('--gpu-metrics-device')), None)
                _nsys_help(help_outputs, metrics_arg)
            result = run(cmd.argv, directory)
            _need(isinstance(result, CommandResult) and type(result.returncode) is int
                  and type(result.stdout) is bytes and type(result.stderr) is bytes, 'bad profiler command result')
            for suffix, blob in [('stdout', result.stdout), ('stderr', result.stderr)]:
                with (directory / f'{cmd.name}.{suffix}').open('xb') as output:
                    output.write(blob)
            log.write((json.dumps(dict(name=cmd.name, argv=cmd.argv, returncode=result.returncode)) + '\n').encode())
            log.flush()
            _need(result.returncode == 0, f'Nsight command failed: {cmd.name}')
            if cmd.name in ('version', 'stats_help', 'export_help', 'profile_help'):
                help_outputs[cmd.name] = result.stdout


def _policy(p):
    _need(_digest(p.engine) and _text(p.worker_id) and _text(p.boot_id) and _text(p.device_uuid) and _map(p.weights) and
          _map(p.loaded_artifacts), 'invalid controller identity pins')
    _need(all(_digest(x) for x in (p.quality_rule_sha256, p.measure_workload_sha256, p.profile_workload_sha256)),
          'invalid controller workload pins')
    _need(_names(p.quality_cases) and p.quality_cases and _names(p.memory_stages) and
          len(p.memory_stages) >= 2 and (p.memory_limit_bytes is None or _uint(p.memory_limit_bytes, 1)), 'invalid controller Judge coverage')
    _need(isinstance(p.scenes, Mapping) and set(p.scenes) == {'W1', 'W2', 'W3', 'W4'} and
          _uint(p.warm_waves, 1) and _uint(p.timed_waves, 1), 'invalid controller timing coverage')
    for scene, shape in p.scenes.items():
        _need(type(shape) is tuple and len(shape) == (4 if scene == 'W4' else 1) and all(
            type(x) is tuple and len(x) == 2 and _uint(x[0], 1) and _uint(x[1], 2) for x in shape),
            'invalid scene shape')
    _need(p.origin in ('offline_fixture', 'hardware_collection') and _names(p.counter_names)
          and p.counter_names and _names(p.profile_phases), 'invalid evidence policy')
    return nsys_collection_plan(p.profile_target, nsys=p.nsys, gpu_metrics_arg=p.gpu_metrics_arg)


def _common(row, op, p):
    _need((row['schema'], row['operation'], row['engine'], row['worker_id'], row['boot_id'],
           row['device_uuid'], row['origin']) == (SCHEMA, op, p.engine, p.worker_id, p.boot_id, p.device_uuid, p.origin),
          'evidence identity/environment mismatch')
    _need(_text(row['run_id']) and _text(row['service_id']) and row['observer_outside_candidate'] is True
          and row['profiler_enabled'] is (op == 'develop_profile'), 'wrong run isolation/instrumentation')
    start, end = row['start_ns'], row['end_ns']
    _need(_uint(start) and _uint(end) and start < end, 'invalid host monotonic run window')


_BINDING = ('schema', 'operation', 'engine', 'worker_id', 'boot_id', 'device_uuid', 'origin', 'run_id',
            'service_id', 'start_ns', 'end_ns')


def _bind(raw, row):
    _need(type(raw) is dict and all(type(raw[k]) is type(row[k]) and raw[k] == row[k] for k in _BINDING),
          'raw observation run binding differs from receipt')
    return raw


def _raw_rows(path, row):
    with path.open('rb') as stream:
        _bind(_json(next(stream, b'')), row)
        return [_json(line) for line in stream]


def _external_identity(row, p, path):
    raw = _bind(_json(path.read_bytes()), row)
    before, after, loaded = (raw[k] for k in ('weights_before', 'weights_after', 'loaded_artifacts'))
    _need(all(_map(x) and x.keys() == p.weights.keys() for x in (before, after)) and
          _map(loaded) and loaded.keys() == p.loaded_artifacts.keys(), 'incomplete external identity observation')
    t0, t1, t2 = (raw[k] for k in ('weights_before_at_ns', 'loaded_at_ns', 'weights_after_at_ns'))
    _need(_text(raw['weight_source']) and _text(raw['loaded_source']) and all(_uint(t) for t in (t0, t1, t2))
          and row['start_ns'] <= t0 < t1 < t2 <= row['end_ns'], 'unbound external artifact observation')
    violations = []
    if before != dict(p.weights) or after != dict(p.weights):
        violations.append('external weight hash differs from pinned weights')
    if loaded != dict(p.loaded_artifacts):
        violations.append('externally observed loaded artifact differs from pinned build')
    return violations


def _quality(row, p, paths):
    _need(row['rule_sha256'] == p.quality_rule_sha256 and row['isolated_output_probe'] is True,
          'unpinned or non-isolated quality assessment')
    if p.sampling_required:
        from .v3_level3_sampling import checked_samples
        checked_samples(_json(paths['sampling.json'].read_bytes()))
    cases = {}
    for case in _raw_rows(paths['isolated-quality.jsonl'], row):
        name, outcome, witness = (case[k] for k in ('case', 'outcome', 'witness'))
        uncertain = ('mixed_tail_absolute' in p.quality_cases and witness=='trajectory-verdict.json'
                     and name in ('mixed_tail_absolute','paired_cancel_numeric','autoregressive_full_vocab')
                     and outcome=='insufficient_evidence')
        _need(name in p.quality_cases and name not in cases and (outcome in ('pass', 'fail') or uncertain) and
              witness in paths and witness not in (*_OBSERVER_RAW, 'isolated-quality.jsonl', 'external-memory.jsonl')
              and paths[witness].stat().st_size > 0, 'invalid isolated quality outcome/witness')
        cases[name] = outcome
    _need(cases.keys() == set(p.quality_cases), 'incomplete independent quality cases')
    samples = _raw_rows(paths['external-memory.jsonl'], row)
    peak, prev = 0, row['start_ns'] - 1
    source = row['memory_source']
    _need(source in ('nvidia_smi', 'nvidia_smi_compute_pid', 'external_device_meter')
          and (p.memory_source is None or source==p.memory_source) and len(samples) == len(p.memory_stages),
          'missing external memory observations')
    witnesses = set()
    for stage, sample in zip(p.memory_stages, samples):
        t, used = sample['at_ns'], sample['used_bytes']
        scope = 'service_compute_pids' if source=='nvidia_smi_compute_pid' else 'exclusive_device'
        _need(sample['stage'] == stage and sample['scope'] == scope and _uint(used) and _uint(t)
              and max(prev + 1, row['start_ns']) <= t <= row['end_ns'], 'invalid external memory sample')
        name = sample['witness']
        _need(name in paths and name not in witnesses and name not in (*_OBSERVER_RAW, 'isolated-quality.jsonl',
              'external-memory.jsonl'), 'missing or reused external device memory witness')
        witnesses.add(name)
        witness = paths[name].read_text(encoding='utf-8-sig')
        if source == 'nvidia_smi':
            observed = parse_nvidia_smi_device_memory(witness, p.device_uuid)
        elif source == 'nvidia_smi_compute_pid':
            raw = _bind(_json(witness),row)
            gpu = parse_nvidia_smi_process_memory(raw['raw_nvml'],p.device_uuid)
            process = raw['service_processes']
            _need(raw['at_ns']==t and type(process) is dict and bool(process) and gpu and
                  all(re.fullmatch(r'[1-9][0-9]{0,9}',pid) and _uint(info['start_ticks'],1) and _uint(info['mount_ns'],1)
                      for pid,info in process.items()) and {str(pid) for pid in gpu} == set(process) and sum(gpu.values())>0,
                  'compute-memory samples lack stable service/PID ownership')
            observed = sum(gpu.values())
        else:
            observed = _bind(_json(witness), row)
            _need(observed['at_ns'] == t and _uint(observed['used_bytes']), 'external device meter sample invalid')
            observed = observed['used_bytes']
        _need(used == observed, 'external device memory disagrees with raw witness')
        peak, prev = max(peak, used), t
    violations = [f'failed isolated output check: {k}' for k in p.quality_cases if cases[k] == 'fail']
    if p.memory_limit_bytes is not None and peak > p.memory_limit_bytes:
        violations.append('observed device memory exceeds preregistered ceiling')
    audit = {}
    if p.memory_limit_bytes is None:
        # Same-workload comparison only; neither absolute values nor deltas enter the score.
        context = dict(rule=p.quality_rule_sha256, stochastic=p.sampling_required, memory_source=source,
                       boot=p.boot_id, gpu=p.device_uuid, stages=p.memory_stages, weights=dict(p.weights))
        audit['weight_observation'] = dict(workload=digest(canonical(context)), weights=dict(p.weights),
            loaded_artifacts=dict(p.loaded_artifacts), memory_source=source,
            sampled_bytes={sample['stage']: sample['used_bytes'] for sample in samples},
            runtime_representation='not_established_by_file_or_memory_observations')
    return violations, dict(**audit, case_outcomes=cases, observed_peak_sample_bytes=peak, memory_source=source,
                            residency_limits=RESIDENCY_LIMITS)


def _measure(row, p, path):
    _need(row['workload_sha256'] == p.measure_workload_sha256 and row['clock'] == 'external_client_monotonic',
          'wrong independent timing workload or clock')
    expected = {(s,i) for s in p.scenes for i in range(p.warm_waves +
                (p.w3_timed_waves if s=='W3' and p.w3_timed_waves else p.timed_waves))}
    seen, rids, windows = set(), set(), []
    inputs, outputs = set(), {}
    request_outputs = {s: {} for s in p.scenes}
    private_ids = {s: {} for s in p.scenes}
    concurrent = []
    w3_requests = {}
    speculative = {s: dict(reported=0, requests=0, accepted=0, proposed=0, verifies=0) for s in p.scenes}
    if p.natural_eos:
        _need(row.get('timing_mode') == 'natural_eos', 'missing registered natural EOS mode')
        if p.chat_corpus_sha256:
            _need(row.get('chat_corpus_sha256') == p.chat_corpus_sha256, 'missing frozen chat input identity')
        for counts in speculative.values():
            counts['reported_output_tokens'] = 0
            counts['natural_stops'] = 0
    if p.measure_input_hashes is not None:
        _need(set(p.measure_input_hashes) == set(p.scenes) and _uint(p.measure_pair) and
              row['pair_index'] == p.measure_pair, 'wrong timing request plan')
    metrics = {s: [] for s in p.scenes}
    waves = _raw_rows(path, row)
    _need(len(waves) == len(expected), 'incomplete timing waves')
    for wave in waves:
        scene, i = wave['scene'], wave['index']
        _need(_uint(i) and (scene, i) in expected - seen and wave['timed'] is (i >= p.warm_waves), 'invalid timing wave')
        seen.add((scene, i))
        shape, reqs = p.scenes[scene], wave['requests']
        _need(type(reqs) is list and len(reqs) == len(shape), 'incomplete wave request coverage')
        starts, firsts, lasts, dones, ttft, tpot, lengths = [], [], [], [], [], [], []
        for lane, (r, (n, out)) in enumerate(zip(reqs, shape)):
            rid = r['rid']
            start, first, last, done = (r[k] for k in ('start_ns', 'first_ns', 'last_ns', 'done_ns'))
            generated = r['output']
            if p.measure_input_lengths is not None:
                right_shape=r['input']==p.measure_input_lengths[scene][i][lane]
            elif p.chat_corpus_sha256 and scene == 'W3':
                from .v3_level3_chat_timing import W3_INPUT_RANGE

                right_shape = W3_INPUT_RANGE[0] <= r['input'] <= W3_INPUT_RANGE[1]
            else:
                right_shape = r['input'] == n
            _need(_text(rid) and rid not in rids and _uint(r['input'], 1) and _uint(generated, 1 if p.natural_eos else 2)
                  and right_shape and _digest(r['sha256'])
                  and r['first_token_unambiguous'] is True and r['stream_complete'] is True, 'invalid timing request')
            if p.natural_eos:
                _need(r['output_cap'] == out and 0 < generated <= out and
                      (r['finish_reason'] == 'stop' or (r['finish_reason'] == 'length' and generated == out)),
                      'invalid natural termination')
            else:
                _need(generated == out, 'fixed-output length mismatch')
            _need(all(_uint(t) for t in (start, first, last, done))
                  and row['start_ns'] <= start < first <= last <= done <= row['end_ns']
                  and (first < last if generated > 1 else first == last), 'invalid timing request window')
            if p.measure_input_hashes is not None:
                prompt = r['input_sha256']
                _need(_digest(prompt) and prompt not in inputs and prompt == p.measure_input_hashes[scene][i][lane],
                      'reused timing prompt or timing input fingerprint mismatch')
                inputs.add(prompt)
                if p.replay_protocol is not None:
                    _need(p.replay_protocol == BOUND_REPLAY and p.natural_eos, 'unknown replay policy')
                    ids = r['observed_ids']
                    _need(type(ids) is list and len(ids) == generated and
                          all(type(t) is int and 0 <= t < 131072 for t in ids) and
                          digest(canonical(ids)) == r['sha256'] and
                          1 not in ids[:-1] and (r['finish_reason'] != 'stop' or ids[-1] == 1) and
                          type(r.get('observed_finish')) is dict and r['observed_finish'].get('type') == r['finish_reason'] and
                          (r['finish_reason'] != 'stop' or (type(r['observed_finish'].get('matched')) is int and
                                                            r['observed_finish']['matched'] == 1)),
                          'stream tokens, finish or digest differ from observed output')
                    if wave['timed']:
                        private_ids[scene][prompt] = ids

                if wave['timed']:
                    outputs[prompt] = [r['sha256'], generated, r['finish_reason']] if p.natural_eos else r['sha256']
                    if p.natural_eos:
                        request_outputs[scene][prompt] = dict(sha256=r['sha256'], tokens=generated,
                                                               finish_reason=r['finish_reason'])
                    if p.chat_corpus_sha256 and scene == 'W3':
                        visible = r['visible_answer_ns']
                        _need(visible is None or (_uint(visible) and first <= visible <= last),
                              'invalid visible-answer timing')
                        w3_requests[prompt] = dict(input_tokens=r['input'], raw_ttft_ms=(first-start)/1e6,
                                                   visible_answer_ttft_ms=(visible-start)/1e6 if visible else None)
                reported = r['reported_speculation']
                if wave['timed']:
                    speculative[scene]['requests'] += 1
                    if p.natural_eos:
                        speculative[scene]['natural_stops'] += r['finish_reason'] == 'stop'
                if reported is not None:
                    _need(type(reported) is dict and set(reported) == {'accepted','proposed','verifies'} and
                          _uint(reported['accepted']) and _uint(reported['proposed'], 1) and
                          _uint(reported['verifies'], 1) and reported['accepted'] <= reported['proposed'],
                          'invalid server-reported speculative counts')
                    if wave['timed']:
                        speculative[scene]['reported'] += 1
                        if p.natural_eos:
                            speculative[scene]['reported_output_tokens'] += generated
                        for key in reported:
                            speculative[scene][key] += reported[key]
            rids.add(rid)
            starts.append(start)
            firsts.append(first)
            lasts.append(last)
            dones.append(done)
            ttft.append((first - start) / 1e6)
            lengths.append(generated)
            if generated > 1:
                tpot.append((last - first) / (generated - 1) / 1e6)
            if p.natural_eos and wave['timed'] and scene == 'W4':
                concurrent.append(dict(wave=i, lane=lane, tokens=generated, finish_reason=r['finish_reason'],
                                       tpot_ms=(last-first)/(generated-1)/1e6 if generated > 1 else None,
                                       ttft_ms=(first-start)/1e6, completed_ms=(done-start)/1e6))
        a, b = min(starts), max(dones)
        if not p.natural_eos:
            _need(len(reqs) == 1 or max(firsts) < min(lasts), 'concurrent decode overlap not observed')
        windows.append((a, b))
        if wave['timed']:
            ttft_ms = median(ttft)
            tpot_ms = median(tpot) if tpot else None
            overall = sum(lengths) * 1e9 / (b-a)
            rule = DEV_SCORE_RULE[scene]
            score = {'decode_tokens_per_s':1000/tpot_ms if tpot_ms else None,
                     'reciprocal_ttft_per_s':1000/ttft_ms,'wave_output_tokens_per_s':overall}[rule]
            numbers = dict(ttft_ms=ttft_ms,tpot_ms=tpot_ms,output_toks_per_s=overall,dev_score=score)
            if p.natural_eos:
                numbers.update(output_tokens=sum(lengths), e2e_ms=median((d-s)/1e6 for s,d in zip(starts,dones)))
            metrics[scene].append(numbers)
    _need(_disjoint(windows), 'timing waves overlap')
    def measured(values):
        present = [value for value in values if value is not None]
        return median(present) if present else None
    data = dict(scenes={scene: {k: measured(w[k] for w in waves) for k in waves[0]} for scene, waves in metrics.items()},
                profiler_enabled=False)
    _need(all(scene['dev_score'] is not None for scene in data['scenes'].values()), 'no observable decode interval')
    if p.natural_eos:
        data['timing_mode'] = 'natural_eos'
        data['timing_workload_sha256'] = p.measure_workload_sha256
        data['timed_request_outputs'] = request_outputs
        if p.replay_protocol:
            data['replay_protocol'] = p.replay_protocol
            data['_private_replay'] = dict(tokens=private_ids, service=row['service_id'], run=row['run_id'],
                                          rids=sorted(rids), warm=sorted({fingerprint for waves in p.measure_input_hashes.values()
                                          for wave in waves[:p.warm_waves] for fingerprint in wave}))
        data['concurrent_requests'] = concurrent
        if p.chat_corpus_sha256:
            data['chat_corpus_sha256'] = p.chat_corpus_sha256
            data['w3_requests'] = w3_requests
    if p.measure_input_hashes is not None:
        for summary in speculative.values():
            summary['accept_rate'] = summary['accepted'] / summary['proposed'] if summary['proposed'] else None
            if p.natural_eos:
                summary['accepted_draft_tokens_per_verify'] = (summary['accepted'] / summary['verifies']
                                                                 if summary['verifies'] else None)
                summary['served_tokens_per_reported_verify'] = (summary['reported_output_tokens'] / summary['verifies']
                                                                 if summary['verifies'] else None)
        data.update(measurement_pair=p.measure_pair, measurement_inputs_sha256=digest(canonical(p.measure_input_hashes)),
                    timed_outputs_sha256=digest(canonical(outputs)), speculation=dict(source='candidate_server_reported_not_verified', timed=speculative))
    return data


def _disjoint(windows):
    ordered = sorted(windows)
    return all(end <= start for (_, end), (start, _) in zip(ordered, ordered[1:]))


def _csv_rows(path, fields):
    """Find the CSV header (some CLI versions emit a text preamble)."""
    rows = csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig'), newline=''), strict=True)
    for row in rows:
        header = [s.strip() for s in row]
        if set(fields) <= set(header):
            _need(len(header) == len(set(header)), 'ambiguous Nsight CSV header')
            break
    else:
        raise EvidenceError('missing Nsight CSV header: ' + path.name)
    for row in rows:
        if row:
            _need(len(row) == len(header), 'malformed Nsight CSV row')
            yield dict(zip(header, row))


def _csv_integer(text):
    _need(re.fullmatch(r'(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)', text.strip()), 'noninteger Nsight CSV observation')
    return int(text.replace(',', ''))


def _stats(paths):
    trace_count = 0
    for row in _csv_rows(paths['cuda_gpu_trace.stdout'], ('Start (ns)', 'Duration (ns)', 'Name')):
        _need(_csv_integer(row['Start (ns)']) >= 0 and _csv_integer(row['Duration (ns)']) > 0
              and _text(row['Name']), 'empty GPU trace observation')
        trace_count += 1
    _need(trace_count > 0, 'empty Nsight GPU trace')
    counts = []
    for name, count in [('cuda_gpu_kern_sum', 'Instances'), ('cuda_api_sum', 'Num Calls')]:
        total = elapsed = 0
        for row in _csv_rows(paths[name + '.stdout'], (count, 'Total Time (ns)', 'Name')):
            calls = _csv_integer(row[count])
            duration = _csv_integer(row['Total Time (ns)'])
            _need(calls > 0 and duration > 0 and _text(row['Name']), 'empty Nsight summary row')
            total += calls
            elapsed += duration
        _need(total > 0, 'empty Nsight stats summary: ' + name)
        counts.append((total, elapsed))
    return trace_count, *counts


def _columns(db, table):
    # Only fixed internal table names are passed here.
    return {r[1] for r in db.execute(f'PRAGMA table_info({table})')}


_LAUNCH = re.compile(r'(?:cuda|cu)(?:Launch(?:Cooperative)?Kernel|GraphLaunch)(?:$|_)')
_SYNC = re.compile(r'(?:cuda|cu)(?:(?:Device|Ctx|Stream|Event)Synchronize|StreamWaitEvent)(?:$|_)')


def _sqlite_trace(path, max_ns, phases, *, allow_zero_api_duration=False):
    result = {}
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA trusted_schema=OFF')
        _need({'id', 'value'} <= _columns(db, 'StringIds'), 'missing Nsight exported string table')
        def usable(table, fields):
            return set(fields.split()) <= _columns(db, table)
        kernels, api = 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME'
        _need(usable(kernels, 'start end correlationId') and usable(api, 'start end correlationId nameId'),
              'missing Nsight CUDA CPU/GPU timeline schema')
        def duration(a, b):
            _need(_uint(a) and _uint(b) and a < b <= max_ns, 'invalid Nsight relative trace interval')
            return b - a
        launches, launch_ns, sync_ns, api_count, sync_count = {}, 0, 0, 0, 0
        api_domains, zero_duration_api_calls = {}, 0
        for table in (api, 'CUPTI_ACTIVITY_KIND_DRIVER'):
            if not usable(table, 'start end correlationId nameId'):
                continue
            domain_n = domain_ns = 0
            for a, b, ident, name in db.execute(f'SELECT t.start,t.end,t.correlationId,s.value FROM {table} t '
                                               'LEFT JOIN StringIds s ON t.nameId=s.id'):
                if allow_zero_api_duration and _uint(a) and _uint(b) and a == b <= max_ns:
                    # Software traces can quantize a short host API call to a
                    # single timestamp. Count it, but credit no elapsed time.
                    zero_duration_api_calls += 1
                    elapsed = 0
                else:
                    elapsed = duration(a, b)
                _need(_text(name), 'unresolved Nsight API name')
                api_count += 1
                domain_n += 1
                domain_ns += elapsed
                if _LAUNCH.match(name):
                    if _uint(ident, 1):
                        launches[ident] = min(a, launches.get(ident, a))
                    launch_ns += elapsed
                if _SYNC.match(name):
                    sync_ns += elapsed
                    sync_count += 1
            api_domains[table] = (domain_n, domain_ns)
        kernel_ns, kernel_count, matched = 0, 0, 0
        for a, b, ident in db.execute(f'SELECT start,end,correlationId FROM {kernels}'):
            kernel_ns += duration(a, b)
            kernel_count += 1
            if _uint(ident, 1) and ident in launches and launches[ident] <= a:
                matched += 1
        _need(kernel_count > 0 and matched > 0 and sync_count > 0,
              'no correlated CUDA launch/kernel or no observed synchronization API')
        result.update(cpu_api_calls=api_count, gpu_kernel_calls=kernel_count, gpu_kernel_duration_ns=kernel_ns,
                      correlated_kernels=matched, launch_api_duration_ns=launch_ns,
                      synchronization_api_duration_ns=sync_ns, synchronization_api_calls=sync_count)
        if allow_zero_api_duration:
            result['zero_duration_api_calls'] = zero_duration_api_calls
        phase_events = {}
        if phases:
            _need(usable('NVTX_EVENTS', 'start end text textId'), 'missing Nsight NVTX phase ranges')
            for name, a, b in db.execute('''SELECT COALESCE(t.text,s.value),t.start,t.end
                    FROM NVTX_EVENTS t LEFT JOIN StringIds s ON t.textId=s.id WHERE t.end IS NOT NULL'''):
                if name in phases:
                    elapsed = duration(a, b)
                    n, ns = phase_events.get(name, (0, 0))
                    phase_events[name] = n + 1, ns + elapsed
            _need(set(phase_events) == set(phases), 'missing preregistered NVTX phase ranges')
        result['nvtx_phases'] = {name: dict(ranges=n, total_duration_ns=ns)
                                  for name, (n, ns) in phase_events.items()}
        counters = {}
        schema_ok = (usable('GPU_METRICS', 'timestamp typeId metricId rawValue') and
                     usable('TARGET_INFO_GPU_METRICS', 'typeId metricId metricName'))
        if schema_ok:
            for name, a, b, n in db.execute('''SELECT i.metricName, MIN(g.timestamp), MAX(g.timestamp),
                    COUNT(g.rawValue) FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS i
                    ON g.typeId=i.typeId AND g.metricId=i.metricId
                    WHERE typeof(g.rawValue) IN ('integer','real') AND abs(g.rawValue) <= 1.7976931348623157e308
                    GROUP BY i.metricName'''):
                _need(_text(name) and _uint(a) and _uint(b) and a <= b <= max_ns and _uint(n, 1),
                      'invalid Nsight hardware counter sample interval')
                counters[name] = n
    return result, schema_ok, counters, api_domains


def _profile(row, p, plan, paths):
    _need(row['workload_sha256'] == p.profile_workload_sha256, 'wrong profiler workload')
    commands = [_json(line) for line in paths['commands.jsonl'].read_bytes().splitlines()]
    _need(len(commands) == len(plan), 'incomplete Nsight capability/run transcript')
    for c, want in zip(commands, plan):
        _need((c['name'], c['argv']) == (want.name, list(want.argv)) and _uint(c['returncode']) and c['returncode'] == 0,
              'unvalidated Nsight command: ' + want.name)
    version = _nsys_help({k: paths[k + '.stdout'].read_bytes() for k in
                          ('version', 'stats_help', 'export_help', 'profile_help')}, p.gpu_metrics_arg)
    _need(paths['trace.nsys-rep'].stat().st_size > 0, 'empty Nsight capture')
    trace, stats_kernels, stats_api = _stats(paths)
    summary, schema_ok, samples, api_domains = _sqlite_trace(
        paths['trace.sqlite'], row['end_ns'] - row['start_ns'], p.profile_phases)
    # Some releases report runtime APIs only, others include driver API rows.
    runtime = api_domains['CUPTI_ACTIVITY_KIND_RUNTIME']
    combined = tuple(map(sum, zip(*api_domains.values())))
    _need(stats_kernels == (summary['gpu_kernel_calls'], summary['gpu_kernel_duration_ns'])
          and trace >= stats_kernels[0] and stats_api in (runtime, combined),
          'Nsight CSV/SQLite event counts or durations disagree')
    counters = row['hardware_counters']
    _need(type(counters) is dict and set(counters) == set(p.counter_names), 'missing preregistered counter availability')
    out = {}
    for name, claim in counters.items():
        if claim['status'] == 'available':
            _need(name in samples and set(claim) == {'status'}, 'counter availability lacks exported samples: ' + name)
            _need(p.gpu_metrics_arg is not None, 'GPU metrics were not enabled for this capture')
            out[name] = {'status': 'available', 'samples': samples[name]}
        else:
            reason = claim['reason']
            _need(claim['status'] == 'unavailable' and reason in COUNTER_UNAVAILABLE and name not in samples
                  and _text(claim['detail']) and claim['source'] in paths, 'unsubstantiated counter unavailability: ' + name)
            if reason in ('permission_denied', 'unsupported'):
                witness = paths[claim['source']]
                _need(witness.suffix in ('.stdout', '.stderr') and witness.stat().st_size > 0 and
                      claim['detail'] in witness.read_text(encoding='utf-8-sig', errors='replace'),
                      'counter platform/permission diagnostic not archived')
            if reason == 'unsupported_export_schema':
                _need(not schema_ok, 'exported GPU metrics schema is supported')
            if reason == 'no_samples':
                _need(schema_ok, 'no_samples requires a recognized counter schema')
            if reason == 'not_requested':
                _need(p.gpu_metrics_arg is None and claim['source'] == 'commands.jsonl', 'GPU metrics were requested')
            out[name] = {'status': 'unavailable', 'reason': reason, 'detail': claim['detail']}
    return dict(nsys_version=version, timeline=summary, hardware_counters=out,
                counter_capability_resolved=all(c.get('reason') != 'not_requested' for c in out.values()),
                trace_artifacts=('trace.nsys-rep', 'trace.sqlite', *(r + '.stdout' for r in REPORTS)))


_OBSERVER_RAW = ('external-identity.json',)


def validate_evidence(root: Path, policy: EvidencePolicy, receipt_sha256: Mapping[str, str]) -> Check:
    """Validate host receipts under <root>/<develop operation>/receipt.json.

    Return an aggregate status and per-operation checks. A malformed, absent,
    unbound or hash-mismatched observation is *insufficient*, never a zero score.
    A verified discrepancy (wrong weight/build, failed probe, or observed memory
    above the policy ceiling) is a violation. A valid slow run stays valid.

    Archives are finalized, access-controlled snapshots; do not validate a tree
    writable by the candidate or one concurrently being collected.
    """
    plan = _policy(policy)
    root = root.resolve()
    verified, outcomes, rows = {}, {}, {}
    for op in OPERATIONS:
        try:
            _need(not (root / op).is_symlink() and not getattr(root / op, 'is_junction', lambda: False)(),
                  'operation archive links not allowed')
            base = root / op
            pin = receipt_sha256.get(op)
            row = _pinned_json(base, 'receipt.json', pin)
            _common(row, op, policy)
            required = _OBSERVER_RAW
            if op == 'develop_quality':
                required += ('isolated-quality.jsonl', 'external-memory.jsonl')
                if policy.sampling_required:
                    required += ('sampling.json',)
            elif op == 'develop_measure':
                required += ('timing.jsonl',)
                if policy.replay_protocol:
                    required += ('observer.stdout',)
                if policy.measure_input_hashes is not None:
                    required += ('timing-inputs.json',)
            else:
                required += ('commands.jsonl', 'trace.nsys-rep', 'trace.sqlite',
                             *(f'{c.name}.{suffix}' for c in plan for suffix in ('stdout', 'stderr')))
            paths = _artifacts(base, row['artifacts'], required)
            violations = _external_identity(row, policy, paths['external-identity.json'])
            if op == 'develop_quality':
                checks, data = _quality(row, policy, paths)
                violations.extend(checks)
            elif op == 'develop_measure':
                if policy.measure_input_hashes is not None:
                    raw_inputs = paths['timing-inputs.json'].read_bytes()
                    _need(len(raw_inputs) <= 2 << 20 and
                          input_fingerprints(_json(raw_inputs)) == policy.measure_input_hashes,
                          'materialized timing request plan differs')
                    if policy.measure_input_lengths is not None:
                        from .v3_level3_chat_timing import length_map
                        _need(length_map(_json(raw_inputs))==policy.measure_input_lengths,'registered per-lane lengths differ')
                if policy.replay_protocol:
                    _need(policy.replay_protocol == BOUND_REPLAY and policy.natural_eos and
                          policy.measure_input_hashes is not None, 'unregistered stream replay policy')
                    witnessed = [_json(line) for line in paths['observer.stdout'].read_bytes().splitlines()]
                    _need(canonical(witnessed) == canonical(_raw_rows(paths['timing.jsonl'], row)),
                          'host record differs from external timing stream')
                data = _measure(row, policy, paths['timing.jsonl'])
            else:
                data = _profile(row, policy, plan, paths)
            rows[op] = row
            status = ('violation' if violations else 'insufficient_evidence'
                      if 'insufficient_evidence' in data.get('case_outcomes',{}).values() else 'valid'
                      if op == 'develop_quality' else 'observed')
            outcomes[op] = Check(status, '; '.join(violations), dict(data, receipt_id=pin, engine=policy.engine))
            verified[op] = (row['start_ns'], row['end_ns'])
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError,
                csv.Error, sqlite3.Error) as exc:
            outcomes[op] = Check('insufficient_evidence', str(exc))
    conflicts = set()
    for i, a in enumerate(OPERATIONS):
        for b in OPERATIONS[i + 1:]:
            if a in rows and b in rows and (rows[a]['run_id'] == rows[b]['run_id'] or
                    rows[a]['service_id'] == rows[b]['service_id'] or not _disjoint([verified[a], verified[b]])):
                conflicts.update((a, b))
    for op in conflicts:
        if outcomes[op].status != 'violation':
            outcomes[op] = Check('insufficient_evidence', 'cross-operation run reuse or time overlap')
    isolation = all(op in rows and op not in conflicts for op in ('develop_measure', 'develop_profile'))
    status = ('violation' if any(c.status == 'violation' for c in outcomes.values()) else
              'valid' if len(rows) == len(OPERATIONS) and not conflicts else 'insufficient_evidence')
    return Check(status, 'one or more operations lack verified evidence' if
                 any(c.status == 'insufficient_evidence' for c in outcomes.values()) else '',
                 dict(operations=outcomes, profile_isolated_from_timing=isolation, origin=policy.origin,
                      residency_limits=RESIDENCY_LIMITS))
