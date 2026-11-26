"""Build changed runtime components against the authenticated parent deployment."""
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from time import monotonic
from uuid import uuid4

from .v3_harness import canonical, digest
from .v3_level3_build import (
    AOT_ROOT, FETCH_SOURCES, TORCH_CMAKE, BuildError, FileHash, Invocation,
    _claim, _need, _recipe, _stream_file, _verify_tree,
)
from .v3_level3_deploy import MAPS, RuntimeFile, stage_component_deployment

DRIVER = Path(__file__).with_name('v3_level3_component_driver.py')
NATIVE_EXT = {'.cu', '.cuh', '.cc', '.cpp', '.c', '.h', '.hpp', '.inl'}
BUILD_TREE = '_ee_cmake_aot'


def _kind(path):
    from .v3_level3_startup import PATH
    if path == PATH: return 'runtime'  # host applies this source-identified declaration, not the wheel
    leaf = PurePosixPath(path)
    if path.startswith(AOT_ROOT + '/csrc/') and leaf.suffix in NATIVE_EXT:
        return 'native'
    if path.startswith('rust/') and leaf.suffix == '.rs' and leaf.name != 'build.rs':
        return 'rust'
    if path.startswith('test/'):
        return 'test'
    if path.startswith(('python/sglang/srt/', 'python/sglang/kernels/')) and not path.startswith(AOT_ROOT + '/'):
        if leaf.suffix in ('.py', '.pyi', '.json') and leaf.name != '__init__.py':
            return 'python'
    return None


def changed_components(request, previous, contract, deployment):
    """Unknown build inputs and modified packaging scripts require the original cold recipe."""
    if request.build_args_json != previous.build_args_json:
        return None
    changes = {}
    deployed = {f.path: f.sha256 for f in deployment.files}
    for name in contract.base.keys() | previous.files.keys() | request.files.keys():
        old, new = previous.files.get(name), request.files.get(name)
        if old == new == contract.base.get(name):
            continue
        kind = _kind(name)
        if kind is None:
            return None
        if old == new:
            continue
        if kind == 'python':
            runtime = name.removeprefix('python/')
            if old is not None and deployed.get(runtime) != digest(old):
                return None
            if old is None and runtime in deployed:
                return None
        changes[name] = kind
    return changes


def _compile_parallelism(recipe_jobs, memory_mib, native):
    # This is builder RAM, not a restriction on the deployed model. CUDA template
    # translation units share the cgroup with tmpfs, the linker and the build tree.
    if native:
        return min(recipe_jobs, max(1, (memory_mib - 24 * 1024) // (24 * 1024))), 1
    return min(recipe_jobs, max(1, memory_mib // (12 * 1024))), 1


def _commands(changes, args, *, nvcc_threads=1):
    steps = {}
    if 'native' in changes.values():
        steps['configure'] = [
            'cmake', '-S', '/work/src/' + AOT_ROOT, '-B', '/work/build/aot', '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=' + args.get('build_type', 'Release'),
            '-DSKBUILD_SABI_COMPONENT=Development.SABIModule', '-DSKBUILD_SABI_VERSION=3.11',
            '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
            '-DENABLE_CCACHE=ON', f'-DSGL_KERNEL_COMPILE_THREADS={nvcc_threads}', '-DTorch_DIR=' + TORCH_CMAKE,
            *(f'-DFETCHCONTENT_SOURCE_DIR_{name}={path}' for name, path in FETCH_SOURCES),
        ]
        targets = ['common_ops_sm90_build', 'common_ops_sm100_build']
        if any(kind == 'native' and not path.startswith(AOT_ROOT + '/csrc/quantization/gguf/')
               for path, kind in changes.items()):
            targets.append('infllm_ops')
        steps['native'] = ['cmake', '--build', '/work/build/aot', '--target', *targets]
    if 'rust' in changes.values():
        steps['rust'] = ['python3', '-I', 'setup.py', 'build_ext', '--build-lib', '/work/lib']
    return steps


@dataclass(frozen=True)
class CompilerCache:
    directory: Path
    entries: tuple[FileHash, ...]
    identity: str
    tree_timestamps: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ComponentReceipt:
    candidate_id: str
    source_sha256: str
    environment_sha256: str
    cache: CompilerCache | None
    mode: str
    status: str


def _read_bundle(bundle, expected, cache_dir, modules, limits, deadline, commands):
    _need(bundle.stat().st_size <= limits.stream_bytes, 'oversized component output')
    records, caches, seen = {}, [], {'module': {}, 'cache': {}}
    totals = dict(module=0, cache=0)
    caps = dict(module=limits.artifact_bytes, cache=limits.cache_bytes)
    with bundle.open('rb') as output:
        _need(output.readline(64) == b'level3-component-v1\n', 'invalid component output header')
        while True:
            if monotonic() >= deadline:
                raise TimeoutError('component extraction deadline')
            line = output.readline(4097)
            _need(0 < len(line) <= 4096 and line.endswith(b'\n'), 'invalid component frame')
            row = json.loads(line)
            if isinstance(row, list) and len(row) == 2 and row[0] == 'durations':
                times = row[1]
                _need(type(times) is dict and set(times) == set(commands) | {
                    'prepare_source_and_dependencies', 'export_modules_and_cache'
                } and all(type(v) is int and 0 <= v <= limits.wall_seconds * 1000 for v in times.values())
                    and not output.read(1), 'invalid component stage durations')
                break
            _need(type(row) is list and len(row) in (3,4) and row[0] in totals, 'invalid component frame')
            lane, name, size = row[:3]
            _need(type(name) is str and type(size) is int and size >= 0, 'invalid component path or size')
            mtime = row[3] if len(row) == 4 else None
            _need(len(row) == 3 or (lane == 'cache' and name.startswith(BUILD_TREE+'/tree/') and
                  type(mtime) is int and 0 < mtime < (1 << 63)), 'invalid compiler tree timestamp')
            _claim(seen[lane], name, 'file')
            totals[lane] += size
            _need(len(seen[lane]) <= limits.files and totals[lane] <= caps[lane], 'component capacity exceeded')
            if lane == 'module':
                _need(name in expected and size >= 20, 'unexpected shared library')
            else:
                _need('native' in commands, 'unexpected compiler cache')
            target = (modules if lane == 'module' else cache_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out:
                pin = _stream_file(output, out, size, deadline)
            if lane == 'module':
                with target.open('rb') as inp:
                    elf = inp.read(20)
                _need(elf[:7] == b'\x7fELF\x02\x01\x01' and elf[16:20] == b'\x03\x00\xb7\x00',
                      'invalid AArch64 shared library')
                records[name] = RuntimeFile(name, size, pin)
            else:
                target.chmod(0o644)
                if mtime is not None:
                    os.utime(target, ns=(mtime,mtime))
                caches.append(FileHash(name, pin, size, 0o644))
    _need(records.keys() == expected, 'missing rebuilt shared library')
    return records, tuple(sorted(caches, key=lambda x: x.path)), times


def build_components(backend, request, previous, parent_native, parent, changes, job, deadline):
    """Host-controlled dispatch, hashing and composition. The Pi guest cannot issue a build recipe."""
    start = monotonic()
    _need(parent_native.environment_sha256 == backend.environment_sha256 and
          parent_native.source_sha256 == backend._source_identity(previous, deadline),
          'component parent build source or environment differs')
    source_hash = backend._source_identity(request, deadline)
    args = json.loads(request.build_args_json)
    _, jobs = _recipe(args, backend.limits)
    jobs, nvcc_threads = _compile_parallelism(jobs, backend.limits.memory_mib,
                                               'native' in changes.values())
    steps = _commands(changes, args, nvcc_threads=nvcc_threads)
    # A different image, recipe, compiler setting or CMake input starts clean.
    tree_key = digest(canonical(dict(environment=backend.environment_sha256, configure=steps.get('configure'),
                                     component_driver=digest(DRIVER.read_bytes()),
                                     build_args=args, jobs=jobs)))
    expected = set()
    paths = {f.path for f in parent.files}
    if 'native' in steps:
        expected.update(MAPS)
        if 'infllm_ops' in steps['native']:
            infllm = {p for p in paths if p.startswith('sgl_kernel/infllm_ops.') and p.endswith('.so')}
            _need(len(infllm) == 1, 'missing parent infllm extension')
            expected.update(infllm)
    if 'rust' in steps:
        rust = {p for p in paths if p.startswith(('sglang/srt/rust_extensions/',
                'sglang/srt/mem_cache/rust_tree_core/mem_cache.')) and p.endswith('.so')}
        _need(rust, 'missing parent Rust extensions')
        expected.update(rust)
    mode = '+'.join(dict.fromkeys(k for k in ('native', 'rust', 'python') if k in changes.values())) or 'source_only'
    cache = getattr(parent_native, 'cache', None)
    parent_cache = cache.identity if cache else None
    durations = {}
    guest_times = {}
    tree_reused = False
    outcome = 'built'
    detail = ''
    with TemporaryDirectory(dir=job, prefix='.components-') as temp:
        root = Path(temp)
        modules = root / 'modules'
        modules.mkdir()
        updates = {}
        if steps:
            guest = root / 'input'
            source = guest / 'source'
            source.mkdir(parents=True)
            for name, data in request.files.items():
                if monotonic() >= deadline:
                    raise TimeoutError('component source staging deadline')
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(request.modes[name])
                os.utime(path, (1700000000, 1700000000))
            for directory in request.directories:
                (source / directory).mkdir(exist_ok=True, parents=True)
            (guest / 'plan.json').write_bytes(json.dumps(steps).encode())
            shutil.copyfile(DRIVER, guest / 'driver.py')
            incoming = guest / 'cache'
            if cache and 'native' in steps:
                _verify_tree(cache.directory, cache.entries, deadline=deadline)
                _need(all((cache.directory/name).stat().st_mtime_ns == stamp for name,stamp in cache.tree_timestamps),
                      'compiler build tree timestamps changed')
                shutil.copytree(cache.directory, incoming, copy_function=os.link)
            elif 'native' in steps and getattr(parent_native, 'cache_output_sha256', None):
                key = digest(canonical([previous.candidate_id, parent_native.source_sha256, backend.environment_sha256]))
                parent_cache = backend._open_cache(backend.root/'cache'/key, incoming, deadline, key)
                _need(parent_cache == parent_native.cache_output_sha256, 'parent compiler cache drift')
            else:
                incoming.mkdir()
            marker = incoming / BUILD_TREE / 'recipe.sha256'
            tree_reused = (marker.is_file() and marker.read_text('ascii') == tree_key and
                           'native' in steps and 'rust' not in steps)
            (guest/'tree-plan.json').write_bytes(json.dumps(dict(
                key=tree_key, reuse=tree_reused, changed=[p for p,k in changes.items() if k == 'native'])).encode())
            # The host may use umask 077. These three trusted inputs are mounted
            # read-only and must be readable by the unrelated builder UID. Do not
            # chmod cached hardlinks, candidate source, the job root, or its journal.
            for metadata in ('driver.py', 'plan.json', 'tree-plan.json'):
                (guest / metadata).chmod(0o444)
            for here, _, _ in os.walk(guest):
                Path(here).chmod(0o755)
            bundle = root / 'bundle'
            invoke = Invocation('ee-native-' + uuid4().hex, guest, deadline, backend.limits, bundle,
                                request.candidate_id, 'incremental', backend.image, (), jobs)
            durations['source_and_cache_staging'] = round(1000*(monotonic()-start))
            began = monotonic()
            result = backend.executor.execute(invoke)
            durations['container_and_transfer'] = round(1000*(monotonic()-began))
            (job/'build.stderr-tail').write_bytes(result.log_tail[-backend.limits.log_bytes:])
            if result.status != 'exited' or result.exit_code != 0:
                outcome = result.status if result.status != 'exited' else 'build_failed'
            else:
                began = monotonic()
                output_cache = root/'output-cache'
                output_cache.mkdir()
                try:
                    updates, entries, guest_times = _read_bundle(bundle, expected, output_cache, modules,
                                                                 backend.limits, deadline, steps)
                    if entries:
                        target = job/'compiler-cache'
                        output_cache.rename(target)
                        timestamps = tuple((e.path, (target/e.path).stat().st_mtime_ns) for e in entries
                                           if e.path.startswith(BUILD_TREE+'/tree/'))
                        identity = digest(canonical(dict(files=[[e.path, e.sha256, e.size] for e in entries],
                                                         tree_timestamps=timestamps)))
                        cache = CompilerCache(target, entries, identity, timestamps)
                except (BuildError, OSError, ValueError) as exc:
                    outcome, detail = 'invalid_output', str(exc)
                durations['validate_modules_and_cache'] = round(1000*(monotonic()-began))
        deployment = None
        if outcome == 'built':
            began = monotonic()
            deleted = set()
            for name, kind in changes.items():
                if kind != 'python':
                    continue
                runtime_path = name.removeprefix('python/')
                if name not in request.files:
                    deleted.add(runtime_path)
                    continue
                data = request.files[name]
                path = modules/runtime_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                updates[runtime_path] = RuntimeFile(runtime_path, len(data), digest(data))
            deployment = stage_component_deployment(parent, request.candidate_id, job.name, job/'deployment',
                                                    modules, updates, deleted)
            durations['deploy_and_verify'] = round(1000*(monotonic()-began))
    native = ComponentReceipt(request.candidate_id, source_hash, backend.environment_sha256, cache, mode, outcome)
    public = dict(build_mode='component_' + mode, build_ms=round(1000*(monotonic()-start)),
                  environment_sha256=backend.environment_sha256, parent_deployment=parent.identity,
                  host_phase_ms=durations, container_reported_phase_ms=guest_times,
                  cmake_tree_reused=tree_reused, compile_jobs=jobs, nvcc_threads=nvcc_threads,
                  cache_in=parent_cache if 'native' in steps else None,
                  cache_out=cache.identity if cache else None, changed_files=len(changes))
    if detail:
        public['detail'] = detail[:300]
    if deployment:
        public['deployment'] = deployment.identity
        public['base_wheels'] = dict(deployment.wheels)
    return native, deployment, public
