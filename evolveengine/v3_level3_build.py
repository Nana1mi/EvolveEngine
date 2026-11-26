"""Level 3 Native build lane. No Worker, deployment, Judge or GPU integration.

The caller is the trusted source contract, not an Agent RPC. Supply its entire
portable effective tree (bytes, modes and directories) and its SHA-256 candidate
ID. The backend also hashes the exact staged tree. Use a dedicated private state
root (one backend owner) on a quota-controlled local filesystem.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zlib
from types import MappingProxyType
from typing import Mapping, Protocol
from uuid import uuid4
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile


SGLANG_REVISION = 'b2ce7f8811bc40fd0d67000030c22edbbe0f2a37'
SGLANG_ARCHIVE_SHA256 = 'e3a71cd04d9513f47e25ffa91644b130b4b20adf0b21554a0b42a78b5433c4ae'
SCHEMA = 'level3-native-build-v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
IMAGE = re.compile(r'(?:[a-z0-9][a-z0-9._/:\-]*@)?sha256:[0-9a-f]{64}\Z')
MIB = 1 << 20
GIB = 1 << 30
MODES = frozenset({'cold', 'incremental', 'independent_cold'})
# Docker 29 can omit unset config fields; use map lookup so absence prints null.
# Both preflight and build still reject an image with declared volumes.
DOCKER_IMAGE_PLATFORM_AND_VOLUMES = '{{.Os}}/{{.Architecture}} {{json (index .Config "Volumes")}}'
# Verified against the historical archive, not current upstream HEAD. The AOT
# wheel distribution is sglang-kernel (import sgl_kernel); python/ builds sglang.
AOT_ROOT = 'python/sglang/kernels/aot'
AOT_GGUF = AOT_ROOT + '/csrc/quantization/gguf/gguf_kernel.cu'
ENGINE_GGUF = 'python/sglang/srt/layers/quantization/gguf.py'
REQUIRED_SOURCE = frozenset({
    'python/pyproject.toml', 'python/setup.py', 'rust/Cargo.toml', 'rust/Cargo.lock',
    AOT_ROOT + '/pyproject.toml', AOT_ROOT + '/CMakeLists.txt', AOT_GGUF, ENGINE_GGUF,
})
# The pinned image must contain immutable extracts of the aot/CMakeLists.txt
# URL_HASH pins (cutlass 57e3cfb..., fmt 553ec11..., flashinfer bc29697...).
# Only the image provisioner can attest how those dirs were obtained. Candidate
# build scripts remain arbitrary code, confined to the offline container.
DEPS_ROOT = '/opt/ee-native'
# Both images contain a system Torch as well as the runtime's pinned Torch.
# Pin CMake directly; an isolated Python import alone does not bind the headers.
TORCH_ROOT = '/opt/spark26-deps/torch'
TORCH_CMAKE = TORCH_ROOT + '/share/cmake/Torch'
FETCH_SOURCES = tuple((f'REPO-{dep.upper()}', f'{DEPS_ROOT}/sources/{dep}')
                      for dep in ('cutlass', 'fmt', 'flashinfer'))
WHEEL = ('python3', '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
         'wheel', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir')
COMMANDS = (
    (*WHEEL, '--config-settings=build-dir=/work/build/aot', '--wheel-dir=/work/artifacts/native',
     '--config-settings=cmake.define.FETCHCONTENT_FULLY_DISCONNECTED=ON',
     '--config-settings=cmake.define.ENABLE_CCACHE=ON',
     f'--config-settings=cmake.define.Torch_DIR={TORCH_CMAKE}',
     '--config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS=1',
     *(f'--config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_{name}={path}' for name, path in FETCH_SOURCES),
     '/work/src/' + AOT_ROOT),
    (*WHEEL, '--wheel-dir=/work/artifacts/engine', '/work/src/python'),
)
BUILD_ENV = (
    'PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/work', 'TMPDIR=/work/tmp',
    'LC_ALL=C.UTF-8', 'TZ=UTC', 'SOURCE_DATE_EPOCH=1700000000',
    'NVIDIA_VISIBLE_DEVICES=void', 'CUDA_VISIBLE_DEVICES=',
    'PIP_NO_INDEX=1', 'PIP_CONFIG_FILE=/dev/null', 'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=/opt/spark26-deps',
    'CARGO_HOME=/work/cargo', 'CARGO_TARGET_DIR=/work/build/rust', 'CARGO_NET_OFFLINE=true',
    'SGLANG_BUILD_RUST_EXTS=all', 'CCACHE_DIR=/work/ccache', 'CCACHE_BASEDIR=/work/src',
    'CCACHE_COMPILERCHECK=content',
)
LINUX_ARM64 = r'(?:linux|manylinux[0-9]+|manylinux_[0-9]+_[0-9]+)_aarch64'
WHEEL_NAME = re.compile(
    rf'(?:native/sglang_kernel-[0-9][A-Za-z0-9_.+]*-cp311-abi3-{LINUX_ARM64}'
    rf'|engine/sglang-[0-9][A-Za-z0-9_.+]*-cp[0-9]+-(?:cp[0-9]+|abi3)-{LINUX_ARM64})\.whl'
)


class BuildError(ValueError):
    """Invalid inputs, configuration, or artifact evidence."""


class CleanupUncertain(RuntimeError):
    """Container cleanup was not proved; quarantine the build backend."""


def _need(ok, message):
    if not ok:
        raise BuildError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _digest(data):
    return sha256(data).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _path(name):
    _need(isinstance(name, str) and 0 < len(name) <= 512 and name == unicodedata.normalize('NFC', name)
          and not any(0xd800 <= ord(c) <= 0xdfff for c in name), 'invalid build path')
    parts = name.split('/')
    _need(len(parts) <= 48 and all(p and p not in {'.', '..'} and p.casefold() != '.git'
          and not p.endswith(('.', ' ')) and not re.fullmatch(
              r'(?i)(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)', p.split('.')[0].rstrip(' '))
          and len(p.encode('utf-8')) <= 255
          and not any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>"|?*' for c in p)
          for p in parts), 'unsafe or nonportable build path')
    return parts


def _claim(seen, name, kind):
    parts = _path(name)
    for i in range(1, len(parts) + 1):
        path = '/'.join(parts[:i])
        tag = kind if i == len(parts) else 'directory'
        _need(seen.setdefault(path.casefold(), (path, tag)) == (path, tag),
              'source/output path collision')


@dataclass(frozen=True)
class BuildRequest:
    candidate_id: str
    files: Mapping[str, bytes]
    modes: Mapping[str, int]
    directories: frozenset[str] = frozenset()
    mode: str = 'cold'
    revision: str = SGLANG_REVISION
    archive_sha256: str = SGLANG_ARCHIVE_SHA256
    build_args_json: bytes = b'{}'

    @classmethod
    def from_contract(cls, contract, engine, *, mode='cold'):
        """Adapt the trusted Level 3 SourceContract; it revalidates the Engine ID."""
        provenance = contract.provenance
        _need(provenance.get('revision') == SGLANG_REVISION
              and provenance.get('archive_sha256') == SGLANG_ARCHIVE_SHA256,
              'missing pinned historical source provenance')
        files, modes, dirs = contract.effective_tree(engine)
        return cls(engine.digest, files, modes, frozenset(dirs), mode,
                   provenance['revision'], provenance['archive_sha256'], _json(engine.build_args))

    def __post_init__(self):
        _need(_is_sha(self.candidate_id), 'candidate ID must be lowercase SHA-256')
        _need(self.revision == SGLANG_REVISION and self.archive_sha256 == SGLANG_ARCHIVE_SHA256,
              'source must come from the pinned historical SGLang archive contract')
        _need(isinstance(self.mode, str) and self.mode in MODES, 'unregistered Native build mode')
        _need(type(self.build_args_json) is bytes and len(self.build_args_json) <= 2048,
              'expected bounded structured build parameters')
        try:
            _need(_json(json.loads(self.build_args_json)) == self.build_args_json,
                  'build parameters are not canonical')
        except (ValueError, RecursionError) as failure:
            raise BuildError('invalid structured build parameters') from failure
        # Take a snapshot; no caller mutation or mutable bytes after admission.
        object.__setattr__(self, 'files', MappingProxyType(dict(self.files)))
        object.__setattr__(self, 'modes', MappingProxyType(dict(self.modes)))
        object.__setattr__(self, 'directories', frozenset(self.directories))


@dataclass(frozen=True)
class BuildLimits:
    wall_seconds: int = 3600
    cpus: int = 8
    memory_mib: int = 24576
    pids: int = 512
    scratch_mib: int = 16384
    source_bytes: int = 512 * MIB
    artifact_bytes: int = 2 * GIB
    cache_bytes: int = 4 * GIB
    files: int = 50_000
    log_bytes: int = 128 * 1024

    def __post_init__(self):
        maxima = (21600, 64, 262144, 4096, 131072, 2 * GIB, 16 * GIB, 32 * GIB, 100_000, MIB)
        _need(all(type(n) is int and 0 < n <= cap for n, cap in zip(asdict(self).values(), maxima))
              and self.memory_mib >= 64 and self.scratch_mib >= 16, 'invalid Native resource limits')

    @property
    def stream_bytes(self):
        # At most 2 * files headers; limit framing as well as file payload.
        return self.artifact_bytes + self.cache_bytes + 2 * self.files * 4096 + 4096


@dataclass(frozen=True)
class Invocation:
    job_id: str
    input_dir: Path
    deadline: float
    limits: BuildLimits
    bundle_path: Path
    candidate_id: str
    mode: str
    image: str
    commands: tuple[tuple[str, ...], ...]
    jobs: int


@dataclass(frozen=True)
class Execution:
    exit_code: int | None
    log_tail: bytes = b''
    status: str = 'exited'


class Executor(Protocol):
    """Trusted adapter; writes invocation.bundle_path and settles cleanup before return."""

    def execute(self, invocation: Invocation) -> Execution: ...


@dataclass(frozen=True)
class FileHash:
    path: str
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class BuildReceipt:
    job_id: str
    candidate_id: str
    source_sha256: str
    environment_sha256: str
    image: str
    mode: str
    status: str
    elapsed_ms: int
    cache_input_sha256: str | None
    cache_output_sha256: str | None
    artifacts: tuple[FileHash, ...] = ()
    artifact_dir: Path | None = None
    exit_code: int | None = None
    log_tail: bytes = b''  # Untrusted diagnostics, not quality or metering evidence.
    detail: str = ''
    limits: BuildLimits = BuildLimits()
    source_revision: str = SGLANG_REVISION
    source_archive_sha256: str = SGLANG_ARCHIVE_SHA256
    build_args_json: bytes = b'{}'
    effective_jobs: int = 0
    executor_kind: str = 'unregistered'

    @property
    def artifact_hashes(self):
        return {entry.path: entry.sha256 for entry in self.artifacts}


@dataclass(frozen=True)
class ColdComparison:
    identical: bool
    differing_paths: tuple[str, ...]


def _stream_file(src, dest, size, deadline=None):
    h = sha256()
    while size:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Native build deadline reached')
        data = src.read(min(size, MIB))
        _need(data, 'incomplete Native build file')
        h.update(data)
        if dest is not None:
            dest.write(data)
        size -= len(data)
    return h.hexdigest()


def _file_hash(path, mode=None, deadline=None):
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and not _reparse(info), 'not a regular build file')
    with path.open('rb') as stream:
        return FileHash('', _stream_file(stream, None, info.st_size, deadline), info.st_size,
                        mode if mode is not None else (0o755 if info.st_mode & 0o111 else 0o644))


def _reparse(info):
    return bool(getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def _verify_tree(root, expected, *, deadline=None):
    pending = [root]
    entries = {item.path: item for item in expected}
    while pending:
        directory = pending.pop()
        info = directory.lstat()
        _need(stat.S_ISDIR(info.st_mode) and not _reparse(info), 'unsafe build evidence directory')
        for file in directory.iterdir():
            info = file.lstat()
            if stat.S_ISDIR(info.st_mode) and not _reparse(info):
                pending.append(file)
                continue
            name = file.relative_to(root).as_posix()
            item = entries.pop(name, None)
            _need(item is not None and info.st_size == item.size, 'unexpected or changed build evidence file')
            actual = _file_hash(file, item.mode if os.name == 'nt' else None, deadline)
            _need((item.sha256, item.mode) == (actual.sha256, actual.mode), 'build evidence SHA-256 or mode mismatch')
    _need(not entries, 'missing build evidence files')


def verify_artifacts(receipt: BuildReceipt) -> None:
    """Rehash saved bytes before a trusted deployer consumes them (does not deploy)."""
    _need(receipt.status == 'built' and receipt.artifact_dir is not None and receipt.artifacts,
          'no verified Native artifacts')
    _verify_tree(receipt.artifact_dir, receipt.artifacts)


def compare_cold_builds(first: BuildReceipt, second: BuildReceipt) -> ColdComparison:
    _need(first.job_id != second.job_id and first.mode in ('cold','independent_cold')
          and second.mode in ('cold','independent_cold') and 'independent_cold' in (first.mode,second.mode)
          and first.cache_input_sha256 is None and second.cache_input_sha256 is None,
          'require distinct cache-free builds including an independent cold build')
    _need((first.candidate_id, first.source_sha256, first.environment_sha256)
          == (second.candidate_id, second.source_sha256, second.environment_sha256),
          'cold builds do not share exact source and environment')
    verify_artifacts(first)
    verify_artifacts(second)
    a, b = ({item.path: item for item in receipt.artifacts} for receipt in (first, second))
    diff = tuple(name for name in sorted(a.keys() | b.keys()) if a.get(name) != b.get(name))
    return ColdComparison(not diff, diff)


# Runs only inside the container. stdout is an untrusted data channel, never a
# host shell or trusted claim of runtime behavior. Redirect build logs to stderr.
_DRIVER = r'''
import json, os, shutil, stat, subprocess, sys
from pathlib import Path

commands = json.loads(sys.argv[1])
limits = json.loads(sys.argv[2])
keep_cache = sys.argv[3] == '1'
work = Path('/work')
os.umask(0o022)
for name in ('tmp', 'build', 'ccache', 'artifacts/native', 'artifacts/engine'):
    (work / name).mkdir(parents=True, exist_ok=True)
shutil.copytree('/ee-input/source', work / 'src')
if not shutil.which('ccache'):
    sys.exit('pinned image is missing ccache')
for relative in ('cutlass/include/cutlass/cutlass.h', 'fmt/include/fmt/format.h', 'flashinfer/csrc/norm.cu'):
    if not (Path('/opt/ee-native/sources') / relative).is_file():
        sys.exit('pinned image is missing offline source: ' + relative)
shutil.copytree('/opt/ee-native/cargo', work / 'cargo')
shutil.copytree('/ee-input/cache', work / 'ccache', dirs_exist_ok=True)
os.environ['CCACHE_MAXSIZE'] = str(max(1, limits['cache_bytes'] >> 20)) + 'M'
for cmd in commands:
    result = subprocess.run(cmd, cwd=work / 'src', stdin=subprocess.DEVNULL,
                            stdout=sys.stderr, check=False)
    if result.returncode:
        sys.exit(1)

def emit(root, lane, cap):
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('not a real output root')
    total = count = 0
    for here, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in dirs:
            if not stat.S_ISDIR((Path(here) / name).lstat().st_mode):
                raise ValueError('not a real build directory')
        for name in sorted(names):
            path = Path(here) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('not a regular build output')
            count += 1
            total += info.st_size
            if count > limits['files'] or total > cap:
                raise ValueError('build output limit exceeded')
            mode = 0o755 if info.st_mode & 0o111 else 0o644
            row = json.dumps([lane, path.relative_to(root).as_posix(), info.st_size, mode]).encode('ascii') + b'\n'
            if len(row) > 4096:
                raise ValueError('output path too long')
            out.write(row)
            with path.open('rb') as stream:
                left = info.st_size
                while left:
                    data = stream.read(min(left, 1 << 20))
                    if not data:
                        raise ValueError('file changed during export')
                    out.write(data)
                    left -= len(data)

out = sys.stdout.buffer
out.write(b'level3-native-build-v1\n')
emit(work / 'artifacts', 'artifact', limits['artifact_bytes'])
if keep_cache:
    emit(work / 'ccache', 'cache', limits['cache_bytes'])
out.write(b'null\n')
out.flush()
'''


class DockerExecutor:
    """Real local Linux Docker adapter; daemon, kernel and pinned image are trusted.

    Requires a private local Docker daemon, enforced cgroups and a preloaded Linux
    ARM64 image with Python (>=3.11), Torch 2.14.1, both PEP 517 backends,
    CMake >=3.31, Ninja, ccache, a CUDA >=13 toolchain with driver stubs, and
    Rust/protoc plus the pinned offline Cargo inputs under /opt/ee-native/cargo.
    The daemon/image must have no embedded secrets, weights, or writable volumes. No credentials or Docker config from the
    current user are passed to Docker or to the build. Deliberately no Dockerfile
    execution, image pull, host shell, bind of existing checkout, or docker.sock
    mount into the container. Container isolation is not a kernel exploit defense.
    """

    def __init__(self, *, binary='/usr/bin/docker', socket='/var/run/docker.sock'):
        _need(all(PurePosixPath(x).is_absolute() and not any(c in str(x) for c in '\x00\r\n')
                  for x in (binary, socket)), 'require local Docker binary and Unix socket')
        self.binary, self.socket = str(binary), str(socket)

    @staticmethod
    def create_args(job: Invocation):
        """Pure argv builder, used by tests and the real adapter."""
        limits = job.limits
        # --mount's comma-delimited value must not permit host path injection.
        src = str(job.input_dir)
        _need(job.input_dir.is_absolute() and not any(c in src for c in ',"\r\n'), 'unsafe trusted Docker bind path')
        return [
            'create', '--name', job.job_id, '--pull=never', '--platform=linux/arm64',
            '--runtime=runc', '--network=none', '--ipc=private', '--cgroupns=private',
            '--user=65532:65532', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--read-only', '--no-healthcheck', '--restart=no', '--log-driver=none',
            '--ulimit=core=0:0', '--shm-size=16m', '--env=NVIDIA_VISIBLE_DEVICES=void', f'--cpus={limits.cpus}',
            f'--memory={limits.memory_mib}m', f'--memory-swap={limits.memory_mib}m',
            f'--pids-limit={limits.pids}',
            '--tmpfs', f'/work:rw,exec,nosuid,nodev,size={limits.scratch_mib}m,nr_inodes={limits.files * 4},mode=1777',
            '--mount', f'type=bind,source={src},target=/ee-input,readonly',
            '--workdir=/work', '--entrypoint=/usr/bin/env', job.image, '-i',
            *BUILD_ENV,
            f'MAX_JOBS={job.jobs}', f'CMAKE_BUILD_PARALLEL_LEVEL={job.jobs}',
            'python3', '-I', '/ee-input/driver.py', _json(job.commands).decode('ascii'),
            _json(asdict(limits)).decode('ascii'), '0' if job.mode == 'independent_cold' else '1',
        ]

    def execute(self, invocation: Invocation) -> Execution:
        _need(sys.platform == 'linux', 'real Native Docker builder requires Linux')
        with tempfile.TemporaryDirectory(prefix='ee-native-docker-config-') as config:
            prefix = [self.binary, '--config', config, '--host', f'unix://{self.socket}']
            env = {'PATH': '/usr/bin:/bin', 'HOME': config, 'LANG': 'C.UTF-8'}

            def remaining():
                seconds = invocation.deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError('Native build deadline reached')
                return min(seconds, 15)

            def ctl(args, *, cleanup=False):
                return subprocess.run([*prefix, *args], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      cwd=config, env=env, check=False,
                                      timeout=15 if cleanup else remaining())

            probe = ctl(['image', 'inspect', '--format', DOCKER_IMAGE_PLATFORM_AND_VOLUMES, invocation.image])
            _need(probe.returncode == 0 and probe.stdout.strip() in (b'linux/arm64 null', b'linux/arm64 {}'),
                  'local pinned ARM64 image unavailable or declares volumes')
            try:
                created = ctl(self.create_args(invocation))
            except subprocess.TimeoutExpired as error:
                # The daemon could still commit the create after its client exited.
                try:
                    ctl(['rm', '-f', invocation.job_id], cleanup=True)
                finally:
                    raise CleanupUncertain('local Docker create timed out; containment state unknown') from error
            if created.returncode:
                return Execution(None, status='create_failed')
            try:
                result = self._attach(prefix, env, config, invocation)
                if result.status != 'exited':
                    return result
                state = ctl(['inspect', '--format', '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}',
                             invocation.job_id])
                status = re.fullmatch(rb'exited ([0-9]{1,3}) (true|false)', state.stdout.strip())
                if state.returncode or not status:
                    return Execution(None, result.log_tail, 'transport_failed')
                exit_code = int(status.group(1))
                if status.group(2) == b'true':
                    return Execution(exit_code, result.log_tail, 'memory_limit')
                if result.exit_code != exit_code:
                    return Execution(exit_code, result.log_tail, 'transport_failed')
                return result
            finally:
                try:
                    removed = ctl(['rm', '-f', invocation.job_id], cleanup=True)
                    if removed.returncode:
                        raise CleanupUncertain('local Docker container removal not confirmed')
                except (OSError, subprocess.SubprocessError) as error:
                    raise CleanupUncertain('local Docker container removal not confirmed') from error

    @staticmethod
    def _attach(prefix, env, cwd, job):
        tail = bytearray()
        failed = threading.Event()
        overflow = threading.Event()
        with job.bundle_path.open('xb') as bundle, subprocess.Popen(
                [*prefix, 'start', '--attach', job.job_id], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd, bufsize=0) as process:
            def drain(stream, output):
                total = 0
                try:
                    with stream:
                        while chunk := stream.read(64 * 1024):
                            if output:
                                total += len(chunk)
                                if total > job.limits.stream_bytes:
                                    overflow.set()
                                    return
                                bundle.write(chunk)
                            else:
                                tail.extend(chunk)
                                if len(tail) > job.limits.log_bytes:
                                    del tail[:-job.limits.log_bytes]
                except OSError:
                    failed.set()

            threads = [threading.Thread(target=drain, args=(stream, output), daemon=True)
                       for stream, output in ((process.stdout, True), (process.stderr, False))]
            for thread in threads:
                thread.start()
            status = 'exited'
            try:
                while process.poll() is None or any(thread.is_alive() for thread in threads):
                    if overflow.is_set() or failed.is_set():
                        status = 'output_limit' if overflow.is_set() else 'transport_failed'
                        break
                    if time.monotonic() >= job.deadline:
                        status = 'timeout'
                        break
                    time.sleep(0.02)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for thread in threads:
                    thread.join(timeout=5)
            if any(thread.is_alive() for thread in threads):
                return Execution(process.returncode, status='transport_failed')
            if overflow.is_set():
                status = 'output_limit'
            elif failed.is_set():
                status = 'transport_failed'
            return Execution(process.returncode, bytes(tail), status)


def _recipe(raw, limits):
    _need(type(raw) is dict and set(raw) <= {'build_type', 'cxx_optimization', 'cuda_optimization',
                                            'cuda_fast_math', 'jobs', 'cuda_architectures'},
          'unsupported structured Native build argument')
    # The pinned CMake file supplies its own -O3, per-target fast math, and
    # explicit -gencode flags (sm_121a for CUDA >=13 on aarch64). These cannot
    # honestly be offered as generic CMAKE_* overrides. Use an explicit,
    # source-contract-tracked CMake edit when testing those policies.
    _need(set(raw) <= {'build_type', 'jobs'},
          'pinned AOT recipe does not support optimization, fast-math, or CUDA architectures overrides; '
          'edit the tracked AOT build sources instead')
    flags = []
    if 'build_type' in raw:
        value = raw['build_type']
        _need(value in ('Release', 'RelWithDebInfo', 'MinSizeRel', 'Debug'), 'invalid build type')
        flags.append(f'--config-settings=cmake.build-type={value}')
    jobs = raw.get('jobs', limits.cpus)
    _need(type(jobs) is int and 0 < jobs <= 128, 'invalid job count')
    return ((*COMMANDS[0][:-1], *flags, COMMANDS[0][-1]), COMMANDS[1]), min(limits.cpus, jobs)


def _manifest(key, records):
    return _json({'cache_key': key, 'files': [asdict(x) for x in records]})


def _unpack(path, roots, job):
    seen = {k: {} for k in roots}
    records = {k: [] for k in roots}
    counts = {k: 0 for k in roots}
    totals = {k: 0 for k in roots}
    caps = {'artifact': job.limits.artifact_bytes, 'cache': job.limits.cache_bytes}
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and not _reparse(info) and info.st_size <= job.limits.stream_bytes,
          'invalid or oversized Native output stream')
    with path.open('rb') as stream:
        _need(stream.readline(64) == (SCHEMA + '\n').encode('ascii'), 'missing Native bundle header')
        while True:
            if time.monotonic() >= job.deadline:
                raise TimeoutError('Native build deadline reached')
            header = stream.readline(4097)
            _need(0 < len(header) <= 4096 and header.endswith(b'\n'), 'invalid build frame header')
            try:
                row = json.loads(header)
            except (ValueError, RecursionError) as error:
                raise BuildError('malformed build output frame') from error
            if row is None:
                _need(not stream.read(1), 'trailing bytes after Native output end')
                break
            _need(isinstance(row, list) and len(row) == 4, 'invalid build frame')
            lane, name, size, mode = row
            _need(isinstance(lane, str) and lane in roots and (lane != 'cache' or job.mode != 'independent_cold')
                  and type(size) is int and size >= 0 and type(mode) is int and mode in (0o644, 0o755),
                  'unexpected Native output frame')
            _claim(seen[lane], name, 'file')
            counts[lane] += 1
            totals[lane] += size
            _need(counts[lane] <= job.limits.files and totals[lane] <= caps[lane], 'Native output resource cap exceeded')
            if lane == 'artifact':
                _need(mode == 0o644 and size > 0 and WHEEL_NAME.fullmatch(name),
                      'unexpected wheel output name or mode')
            target = roots[lane].joinpath(*_path(name))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open('xb') as output:
                    digest = _stream_file(stream, output, size, job.deadline)
            except FileExistsError as error:
                raise BuildError('duplicate Native output path') from error
            target.chmod(mode)
            records[lane].append(FileHash(name, digest, size, mode))
    artifacts = records['artifact']
    _need(len(artifacts) == 2 and {x.path.split('/')[0] for x in artifacts} == {'engine', 'native'},
          'require exactly one SGLang wheel and one Native kernel wheel')
    for item in artifacts:
        if time.monotonic() >= job.deadline:
            raise TimeoutError('Native build deadline reached validating wheels')
        _verify_wheel(roots['artifact'] / item.path, item, job.limits.files)
    return tuple(sorted(artifacts, key=lambda x: x.path)), tuple(sorted(records['cache'], key=lambda x: x.path))


# Check bounded metadata and two small ELF prefixes, never extract or import a
# candidate wheel on the host. These are packaging checks, not deploy/load or
# execution evidence. ZIP64 and commented wheels are not emitted by this lane.
def _verify_wheel(path, item, file_cap):
    with path.open('rb') as stream:
        _need(item.size >= 22, 'invalid wheel archive')
        stream.seek(-min(item.size, 42), os.SEEK_END)
        tail = stream.read(42)
        sig, disk, cd_disk, here, count, size, start, comment = struct.unpack('<4s4H2IH', tail[-22:])
        _need(sig == b'PK\x05\x06' and tail[-42:-38] != b'PK\x06\x07' and not (disk or cd_disk or comment)
              and 0 < here == count < 65535 and count <= file_cap and 46 * count <= size <= 16 * MIB
              and start + size + 22 == item.size, 'invalid or unbounded wheel directory')
        try:
            with ZipFile(stream) as wheel:
                infos = wheel.infolist()
                _need(len(infos) == count, 'invalid wheel member count')
                seen, names = {}, set()
                for member in infos:
                    name = member.orig_filename
                    kind = 'directory' if name.endswith('/') else 'file'
                    name = name[:-1] if kind == 'directory' else name
                    _need(name not in names, 'duplicate wheel member')
                    names.add(name)
                    _claim(seen, name, kind)
                    _need(member.compress_type in (ZIP_STORED, ZIP_DEFLATED) and not member.flag_bits & 1
                          and (member.create_system != 3 or
                          stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFDIR if kind == 'directory'
                                                                      else stat.S_IFREG)), 'unsafe wheel member')
                files = {n for n, kind in seen.values() if kind == 'file'}
                dist = item.path.split('/')[1].split('-')[:2]
                meta = '-'.join(dist) + '.dist-info/'
                _need({meta + name for name in ('METADATA', 'WHEEL', 'RECORD')} <= files, 'missing wheel package metadata')
                if item.path.startswith('native/'):
                    _need({'sgl_kernel/__init__.py', 'sgl_kernel/quantization/gguf.py'} <= files,
                          'missing sgl_kernel package or GGUF wrapper')
                    for variant in ('sm90', 'sm100'):
                        name = f'sgl_kernel/{variant}/common_ops.abi3.so'
                        _need(name in files, f'missing pinned AOT {variant} GGUF extension')
                        with wheel.open(name) as library:
                            elf = library.read(20)
                        _need(elf[:7] == b'\x7fELF\x02\x01\x01' and elf[16:20] == b'\x03\x00\xb7\x00',
                              'expected ELF64 little-endian AArch64 shared library')
                else:
                    _need({'sglang/__init__.py', ENGINE_GGUF.removeprefix('python/')} <= files,
                          'missing pinned Engine package or GGUF loader')
        except (BadZipFile, EOFError, NotImplementedError, RuntimeError, ValueError, zlib.error) as error:
            if isinstance(error, BuildError):
                raise
            raise BuildError('invalid wheel archive') from error


class NativeBuildBackend:
    """Submit a multifeature full tree to a fixed offline entrypoint.

    ``cold`` has no cache input and can seed a warm run; ``incremental`` can
    reuse AOT compiler objects through ccache, keyed by exact candidate, tree
    and environment. Each run reconfigures CMake and packages fresh wheels;
    prebuilt wheels and CMake binary trees are never restored. The receipt
    reports cache input, not a measured hit. ``independent_cold`` starts with
    an empty ccache and never publishes one.
    Each builds a new private container/workspace. Store/receipts are for the
    trusted controller; no deployment or live A4/A5 credit is implied.
    """

    def __init__(self, state_root: Path, *, image: str, limits: BuildLimits | None = None,
                 executor: Executor | None = None):
        _need(isinstance(image, str) and IMAGE.fullmatch(image), 'require a pinned OCI digest or immutable local image ID')
        self.root = Path(state_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.image, self.limits = image, limits if limits is not None else BuildLimits()
        _need(isinstance(self.limits, BuildLimits), 'expected Native resource lease')
        self.executor = executor if executor is not None else DockerExecutor()
        self.executor_kind = 'local_docker' if type(self.executor) is DockerExecutor else 'injected'
        self.environment_sha256 = _digest(_json({
            'schema': SCHEMA, 'image': image, 'platform': 'linux/arm64', 'limits': asdict(self.limits),
            'executor_kind': self.executor_kind,
            'driver_sha256': _digest(_DRIVER.encode('ascii')), 'backend_sha256': _digest(Path(__file__).read_bytes()),
            'fixed_argv': COMMANDS, 'build_env': BUILD_ENV,
        }))
        self.quarantined = False
        self._lock = threading.Lock()
        for name in ('ids', 'cache', 'artifacts'):
            (self.root / name).mkdir(exist_ok=True)

    def build(self, request: BuildRequest, *, deadline: float | None = None) -> BuildReceipt:
        _need(isinstance(request, BuildRequest), 'require a trusted full-source BuildRequest')
        if deadline is not None:
            _need(type(deadline) in (int, float) and deadline > time.monotonic(), 'invalid or expired build lease')
        with self._lock:
            _need(not self.quarantined and not (self.root / 'quarantined').exists(),
                  'Native builder quarantined: verify cleanup before another dispatch')
            started = time.monotonic()
            stop = min(deadline, started + self.limits.wall_seconds) if deadline is not None else started + self.limits.wall_seconds
            return self._build(request, started, stop)

    def _build(self, request, started, deadline):
        commands, jobs = _recipe(json.loads(request.build_args_json), self.limits)
        source_hash = self._source_identity(request, deadline)
        binding = self.root / 'ids' / request.candidate_id
        if binding.exists():
            _need(binding.read_bytes() == source_hash.encode('ascii'), 'candidate ID is already bound to different effective input')
        else:
            binding.write_bytes(source_hash.encode('ascii'))
        cache_key = _digest(_json([request.candidate_id, source_hash, self.environment_sha256]))
        cached = self.root / 'cache' / cache_key
        job_id = 'ee-native-' + uuid4().hex
        cache_in = None
        outcome = Execution(None)

        def receipt(status, *, artifacts=(), artifact_dir=None, cache_out=None, detail=''):
            return BuildReceipt(job_id, request.candidate_id, source_hash, self.environment_sha256, self.image,
                                request.mode, status, max(0, int((time.monotonic() - started) * 1000)),
                                cache_in, cache_out, artifacts, artifact_dir, outcome.exit_code,
                                outcome.log_tail[-self.limits.log_bytes:], detail, self.limits,
                                build_args_json=request.build_args_json, effective_jobs=jobs, executor_kind=self.executor_kind)

        with tempfile.TemporaryDirectory(prefix='.native-attempt-', dir=self.root) as scratch:
            stage = Path(scratch)
            guest = stage / 'input'
            guest.mkdir(mode=0o755)
            (guest / 'driver.py').write_bytes(_DRIVER.encode('ascii'))
            (guest / 'driver.py').chmod(0o644)
            source = guest / 'source'
            source.mkdir(mode=0o755)
            cache = guest / 'cache'
            job = Invocation(job_id, guest, deadline, self.limits, stage / 'bundle', request.candidate_id,
                             request.mode, self.image, commands, jobs)
            try:
                if request.mode == 'incremental' and cached.exists():
                    cache_in = self._open_cache(cached, cache, deadline, cache_key)
                else:
                    cache.mkdir(mode=0o755)
                for name in sorted(request.directories):
                    source.joinpath(*_path(name)).mkdir(mode=0o755, parents=True, exist_ok=True)
                for name, data in sorted(request.files.items()):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Native build deadline reached during staging')
                    target = source.joinpath(*_path(name))
                    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                    target.write_bytes(data)
                    target.chmod(request.modes[name])
                    os.utime(target, (1700000000, 1700000000))
                # The enclosing temporary/state directories remain private; only
                # this bind root and its descendants are visible to the non-root UID.
                for here, _, _ in os.walk(guest):
                    Path(here).chmod(0o755)
                outcome = self.executor.execute(job)
            except CleanupUncertain as failure:
                self.quarantined = True
                self.root.joinpath('quarantined').write_text(job_id, encoding='ascii')
                return receipt('cleanup_uncertain', detail=str(failure))
            except (TimeoutError, subprocess.TimeoutExpired) as failure:
                return receipt('timeout', detail=str(failure))
            except (BuildError, OSError, subprocess.SubprocessError) as failure:
                return receipt('executor_failed', detail=str(failure))
            if outcome.status != 'exited':
                return receipt(outcome.status)
            if outcome.exit_code != 0:
                return receipt('build_failed')
            try:
                art, next_cache = stage / 'artifacts', stage / 'next-cache'
                art.mkdir(); next_cache.mkdir()
                (next_cache / 'files').mkdir()
                artifacts, entries = _unpack(job.bundle_path, {'artifact': art, 'cache': next_cache / 'files'}, job)
                _verify_tree(art, artifacts, deadline=deadline)
                cache_out = None
                if request.mode != 'independent_cold':
                    _verify_tree(next_cache / 'files', entries, deadline=deadline)
                    data = _manifest(cache_key, entries)
                    (next_cache / 'manifest.json').write_bytes(data)
                    old = stage / 'old-cache'
                    if cached.exists():
                        cached.replace(old)
                    try:
                        next_cache.replace(cached)
                    except OSError:
                        if old.exists():
                            old.replace(cached)
                        raise
                    cache_out = _digest(data) if entries else None
                saved = self.root / 'artifacts' / job_id
                art.replace(saved)
                return receipt('built', artifacts=artifacts, artifact_dir=saved, cache_out=cache_out)
            except TimeoutError as failure:
                return receipt('timeout', detail=str(failure))
            except (BuildError, OSError) as failure:
                return receipt('invalid_output', detail=str(failure))

    def _source_identity(self, request, deadline):
        _need(request.files and set(request.files) == set(request.modes), 'require a complete source/mode manifest')
        _need(REQUIRED_SOURCE <= request.files.keys(), 'missing pinned AOT or Engine build inputs')
        _need(len(request.files) + len(request.directories) <= self.limits.files, 'source inventory exceeds count cap')
        items, seen, total = [], {}, 0
        _need(all(isinstance(p, str) for p in (*request.directories, *request.files)), 'non-text source path')
        for name in sorted(request.directories):
            _claim(seen, name, 'directory')
            _need(len(seen) <= self.limits.files, 'source inventory exceeds count cap')
        for name, data in sorted(request.files.items()):
            if time.monotonic() >= deadline:
                raise TimeoutError('Native build deadline reached checking source')
            _claim(seen, name, 'file')
            _need(type(data) is bytes and type(request.modes[name]) is int and request.modes[name] in (0o644, 0o755),
                  'source contract must supply exact immutable bytes and normalized mode')
            total += len(data)
            _need(len(seen) <= self.limits.files and total <= self.limits.source_bytes, 'source exceeds resource cap')
            items.append([name, _digest(data), len(data), request.modes[name]])
        directories = sorted(path for path, kind in seen.values() if kind == 'directory')
        return _digest(_json([request.revision, request.archive_sha256, directories, items, json.loads(request.build_args_json)]))

    def _open_cache(self, root, target, deadline, cache_key):
        _need(stat.S_ISDIR(root.lstat().st_mode) and not _reparse(root.lstat()), 'unsafe candidate cache root')
        lock = root / 'manifest.json'
        info = lock.lstat()
        _need(stat.S_ISREG(info.st_mode) and not _reparse(info) and info.st_size <= self.limits.files * 4096 + 4096,
              'invalid candidate cache manifest')
        raw = lock.read_bytes()
        try:
            data = json.loads(raw)
            _need(type(data) is dict and data.keys() == {'cache_key', 'files'}
                  and data['cache_key'] == cache_key and type(data['files']) is list
                  and len(data['files']) <= self.limits.files, 'wrong or unbounded candidate cache manifest')
            entries = tuple(FileHash(**item) for item in data['files'])
        except (TypeError, ValueError, RecursionError) as error:
            raise BuildError('invalid candidate cache manifest') from error
        seen = {}
        for item in entries:
            _claim(seen, item.path, 'file')
            _need(_is_sha(item.sha256) and type(item.size) is int and item.size >= 0 and item.mode in (0o644, 0o755),
                  'invalid candidate cache manifest entry')
        _need(sum(item.size for item in entries) <= self.limits.cache_bytes and _manifest(cache_key, entries) == raw,
              'non-canonical candidate cache manifest')
        _verify_tree(root / 'files', entries, deadline=deadline)
        shutil.copytree(root / 'files', target)
        _verify_tree(target, entries, deadline=deadline)
        return _digest(raw) if entries else None
