from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from evolveengine.v3_level3_build import (
    AOT_GGUF, AOT_ROOT, ENGINE_GGUF, BuildError, BuildLimits, BuildRequest, CleanupUncertain, DockerExecutor,
    Execution, Invocation, NativeBuildBackend, SCHEMA, SGLANG_ARCHIVE_SHA256,
    SGLANG_REVISION, compare_cold_builds, verify_artifacts,
)

IMAGE = 'evolveengine/native@sha256:' + 'd' * 64
NATIVE_NAME = 'native/sglang_kernel-0.4.11-cp311-abi3-linux_aarch64.whl'
ENGINE_NAME = 'engine/sglang-0.0.0.dev0-cp312-cp312-linux_aarch64.whl'
NATIVE_CPP = AOT_ROOT + '/csrc/common_extension.cc'
FILES = {
    'python/pyproject.toml': b'[build-system]\n',
    'python/setup.py': b'# engine build hook\n',
    'rust/Cargo.toml': b'[workspace]\n',
    'rust/Cargo.lock': b'# pinned Cargo graph\n',
    ENGINE_GGUF: b'# Python GGUF dispatch\n',
    AOT_ROOT + '/pyproject.toml': b'[build-system]\n',
    AOT_ROOT + '/CMakeLists.txt': b'# AOT recipe\n',
    AOT_ROOT + '/python/sgl_kernel/quantization/gguf.py': b'# Native GGUF wrapper\n',
    NATIVE_CPP: b'int answer() { return 7; }\n',
    AOT_GGUF: b'__global__ void f() {}\n',
    'tools/compile.sh': b'#!/bin/sh\nexit 0\n',
    'docs/full-tree.bin': b'\x00\xff',
}
# Minimal ZIPs with pinned package layout and an ELF prefix, not executable
# libraries or pip-installable artifacts. Stable contents for digest comparisons.
SM100 = 'sgl_kernel/sm100/common_ops.abi3.so'
ELF = b'\x7fELF\x02\x01\x01' + bytes(9) + b'\x03\x00\xb7\x00'
NATIVE_MEMBERS = {
    'sgl_kernel/__init__.py': b'# stub',
    'sgl_kernel/quantization/gguf.py': b'# stub',
    'sgl_kernel/sm90/common_ops.abi3.so': ELF,
    SM100: ELF,
}
ENGINE_MEMBERS = {'sglang/__init__.py': b'# stub', ENGINE_GGUF.removeprefix('python/'): b'# overlay stub'}


def wheel_bytes(name, members):
    buffer = BytesIO()
    dist = '-'.join(name.split('/')[1].split('-')[:2]) + '.dist-info/'
    with ZipFile(buffer, 'w') as wheel:
        for path, data in {**members, **{dist + k: b'metadata' for k in ('METADATA', 'WHEEL', 'RECORD')}}.items():
            wheel.writestr(ZipInfo(path), data)
    return buffer.getvalue()


WHEELS = {NATIVE_NAME: wheel_bytes(NATIVE_NAME, NATIVE_MEMBERS), ENGINE_NAME: wheel_bytes(ENGINE_NAME, ENGINE_MEMBERS)}
CACHE_ENTRY = 'a/0/abc123R'

SMALL = BuildLimits(wall_seconds=10, cpus=2, memory_mib=256,
                    scratch_mib=32, source_bytes=4096, artifact_bytes=4096,
                    cache_bytes=4096, files=100, log_bytes=16)


def request(*, files=FILES, mode='cold', label='A', **changes):
    return BuildRequest(sha256(label.encode()).hexdigest(), files,
                        {path: 0o755 if path.endswith('.sh') else 0o644 for path in files},
                        frozenset({'empty-dir'}), mode, **changes)


def put_bundle(path, artifacts, cache=None, *, extra_frames=(), end=b'null\n'):
    with path.open('xb') as spool:
        spool.write(SCHEMA.encode('ascii') + b'\n')
        for lane, files in (('artifact', artifacts), ('cache', cache or {})):
            for name, data in files.items():
                spool.write(json.dumps([lane, name, len(data), 0o644]).encode() + b'\n' + data)
        for header, data in extra_frames:
            spool.write(json.dumps(header).encode() + b'\n' + data)
        spool.write(end)


class FakeExecutor:
    """No subprocess, OS container, native code, network, or GPU is used."""
    def __init__(self):
        self.calls = []
        self.wheels = WHEELS.copy()
        self.cache = {CACHE_ENTRY: b'cache-v1', 'a/0/stats': b'derived-bytes'}
        self.result = Execution(0, b'compile diagnostic with a bounded retained tail')
        self.failure = None
        self.extra_frames = ()
        self.end = b'null\n'

    def execute(self, job):
        root = job.input_dir
        tree = {p.relative_to(root / 'source').as_posix(): p.read_bytes()
                for p in (root / 'source').rglob('*') if p.is_file()}
        cache = {p.relative_to(root / 'cache').as_posix(): p.read_bytes()
                 for p in (root / 'cache').rglob('*') if p.is_file()}
        self.calls.append((job, tree, cache, (root / 'source' / 'empty-dir').is_dir(),
                           set(p.name for p in root.iterdir())))
        if self.failure:
            raise self.failure
        if self.result.exit_code == 0:
            put_bundle(job.bundle_path, self.wheels,
                       None if job.mode == 'independent_cold' else self.cache,
                       extra_frames=self.extra_frames, end=self.end)
        return self.result


class NativeBackendTests(unittest.TestCase):
    def test_aot_uses_runtime_torch_cmake_config(self):
        from evolveengine.v3_level3_build import BUILD_ENV, COMMANDS
        self.assertIn('--config-settings=cmake.define.Torch_DIR=/opt/spark26-deps/torch/share/cmake/Torch',COMMANDS[0])
        self.assertIn('PYTHONPATH=/opt/spark26-deps',BUILD_ENV)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fake = FakeExecutor()
        self.backend = NativeBuildBackend(self.root, image=IMAGE, executor=self.fake, limits=SMALL)

    def test_cold_incremental_independent_and_other_candidate(self):
        a = request()
        cold = self.backend.build(a)
        self.assertEqual(cold.status, 'built')
        self.assertEqual(cold.candidate_id, a.candidate_id)
        self.assertEqual(cold.source_revision, SGLANG_REVISION)
        self.assertEqual(cold.source_archive_sha256, SGLANG_ARCHIVE_SHA256)
        self.assertIsNone(cold.cache_input_sha256)
        self.assertEqual(cold.artifact_hashes, {name: sha256(raw).hexdigest() for name, raw in WHEELS.items()})
        self.assertEqual(cold.log_tail, self.fake.result.log_tail[-16:])
        warm = self.backend.build(replace(a, mode='incremental'))
        separate = self.backend.build(replace(a, mode='independent_cold'))
        b = request(files={**FILES, AOT_GGUF: b'changed native'}, label='B', mode='incremental')
        different = self.backend.build(b)
        self.assertEqual(warm.cache_input_sha256, cold.cache_output_sha256)
        self.assertTrue(warm.cache_input_sha256)
        self.assertIsNone(separate.cache_input_sha256)
        self.assertIsNone(separate.cache_output_sha256)
        self.assertIsNone(different.cache_input_sha256)
        self.assertNotEqual(different.source_sha256, cold.source_sha256)
        self.assertEqual(self.fake.calls[0][1:], (FILES, {}, True, {'driver.py', 'source', 'cache'}))
        self.assertEqual(self.fake.calls[1][2], self.fake.cache)
        self.assertEqual(self.fake.calls[2][2], {})
        self.assertEqual(self.fake.calls[3][1], dict(b.files))
        self.assertEqual(len({job.input_dir for job, *_ in self.fake.calls}), 4)
        self.assertEqual(len({r.artifact_dir for r in (cold, warm, separate, different)}), 4)
        self.assertTrue(compare_cold_builds(cold, separate).identical)
        another = self.backend.build(replace(a, mode='independent_cold'))
        self.assertTrue(compare_cold_builds(separate, another).identical)
        with self.assertRaises(BuildError):
            compare_cold_builds(separate, separate)
        self.assertEqual(self.backend.build(replace(a, mode='incremental')).cache_input_sha256,
                         warm.cache_output_sha256)

    def test_failed_warm_does_not_publish_output_replace_cache_or_qualify(self):
        a = request()
        cold = self.backend.build(a)
        self.fake.result = Execution(7, b'compiler nonzero')
        bad = self.backend.build(replace(a, mode='incremental'))
        self.assertEqual((bad.status, bad.exit_code), ('build_failed', 7))
        self.assertIsNone(bad.artifact_dir)
        self.assertIsNone(bad.cache_output_sha256)
        self.assertEqual(bad.cache_input_sha256, cold.cache_output_sha256)
        self.assertEqual(bad.artifact_hashes, {})
        with self.assertRaises(BuildError):
            verify_artifacts(bad)
        self.fake.result = Execution(0)
        later = self.backend.build(replace(a, mode='incremental'))
        self.assertEqual(later.cache_input_sha256, cold.cache_output_sha256)

    def test_source_hash_binds_same_id_across_modes_and_backend_instances(self):
        cold = self.backend.build(request())
        another = NativeBuildBackend(self.root, image=IMAGE, executor=self.fake, limits=SMALL)
        for bad in (
            request(files={**FILES, 'docs/full-tree.bin': b'other'}),
            replace(request(), modes={**request().modes, 'tools/compile.sh': 0o644}),
            replace(request(), directories=frozenset({'different-empty-dir'})),
            replace(request(), build_args_json=b'{"jobs":1}'),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(BuildError, 'already bound'):
                another.build(bad)
        self.assertEqual(len(self.fake.calls), 1)
        reused = another.build(request(mode='incremental'))
        self.assertEqual(reused.cache_input_sha256, cold.cache_output_sha256)

    def test_different_pinned_image_cannot_read_another_environment_cache(self):
        cold = self.backend.build(request())
        other = NativeBuildBackend(self.root, image=IMAGE[:-1] + 'e', executor=self.fake, limits=SMALL)
        new = other.build(request(mode='incremental'))
        self.assertNotEqual(new.environment_sha256, cold.environment_sha256)
        self.assertIsNone(new.cache_input_sha256)
        self.assertEqual(self.fake.calls[-1][2], {})

    def test_cold_comparison_reports_bytes_and_rejects_tampering(self):
        cold = self.backend.build(request())
        name = next(iter(self.fake.wheels))
        self.fake.wheels[name] = wheel_bytes(name, {**NATIVE_MEMBERS, SM100: ELF + b'changed'})
        other = self.backend.build(request(mode='independent_cold'))
        self.assertEqual(compare_cold_builds(cold, other).differing_paths, (name,))
        with self.assertRaisesRegex(BuildError, 'cache-free'):
            compare_cold_builds(cold, cold)
        (other.artifact_dir / name).write_bytes(bytes(len(self.fake.wheels[name])))  # same length
        with self.assertRaisesRegex(BuildError, 'mismatch'):
            compare_cold_builds(cold, other)

    def test_corrupt_cache_never_reaches_executor(self):
        self.backend.build(request())
        data = next((self.root / 'cache').glob('*/files/' + CACHE_ENTRY))
        data.write_bytes(b'cache-X1')
        warm = self.backend.build(request(mode='incremental'))
        self.assertEqual(warm.status, 'executor_failed')
        self.assertIn('mismatch', warm.detail)
        self.assertEqual(len(self.fake.calls), 1)

    def test_rejects_malformed_bounded_output_before_publication(self):
        a = request()
        cases = (
            ((['cache', '../escaped', 1, 420], b'x'),),
            ((['cache', '/absolute', 1, 420], b'x'),),
            ((['cache', 'NUL.txt', 1, 420], b'x'),),
            ((['cache', 'A/other.o', 1, 420], b'x'),),
            ((['cache', CACHE_ENTRY, 1, 420], b'x'),),
            ((['cache', 'big', SMALL.cache_bytes, 420], b''),),
            ((['cache', 'unknown-mode', 0, 0o4755], b''),),
            ((['artifact', 'some-tool.py', 1, 420], b'x'),),
        )
        for frames in cases:
            with self.subTest(frames=frames):
                self.fake.extra_frames = frames
                r = self.backend.build(a)
                self.assertEqual(r.status, 'invalid_output', r)
                self.assertFalse(r.artifacts)
        self.fake.extra_frames = ()
        self.fake.end = b'null\ntrailing'
        self.assertEqual(self.backend.build(a).status, 'invalid_output')
        self.fake.end = b'null\n'
        self.fake.wheels.pop(next(iter(self.fake.wheels)))
        self.assertEqual(self.backend.build(a).status, 'invalid_output')
        self.assertFalse(list((self.root / 'artifacts').iterdir()))
        self.assertFalse(list((self.root / 'cache').iterdir()))

    def test_source_admission_and_command_injection_rejected_before_execution(self):
        with self.assertRaisesRegex(BuildError, 'historical'):
            request(revision='f' * 40)
        with self.assertRaisesRegex(BuildError, 'SHA-256'):
            replace(request(), candidate_id='not-a-hash')
        with self.assertRaisesRegex(BuildError, 'pinned OCI digest or immutable local image ID'):
            NativeBuildBackend(self.root, image='native:latest', executor=self.fake)
        for path in ('../x', '.GIT/config', AOT_ROOT + '/Csrc/alias.cu', 'data:ads'):
            bad = request(files={**FILES, path: b'input'})
            with self.subTest(path=path), self.assertRaises(BuildError):
                self.backend.build(bad)
        with self.assertRaisesRegex(BuildError, 'structured'):
            self.backend.build(request(build_args_json=b'{"cmd":"/bin/sh"}'))
        with self.assertRaisesRegex(BuildError, 'architectures'):
            self.backend.build(request(build_args_json=b'{"cuda_architectures":["121;whoami"]}'))
        self.assertFalse(self.fake.calls)

    def test_executor_timeout_and_quarantine_are_distinct(self):
        self.fake.failure = TimeoutError('injected wall expiry')
        self.assertEqual(self.backend.build(request()).status, 'timeout')
        self.fake.failure = CleanupUncertain('cannot confirm stopped')
        self.assertEqual(self.backend.build(request()).status, 'cleanup_uncertain')
        self.assertTrue(self.backend.quarantined)
        another = NativeBuildBackend(self.root, image=IMAGE, executor=self.fake, limits=SMALL)
        with self.assertRaisesRegex(BuildError, 'quarantined'):
            another.build(request(label='new'))
        self.assertEqual(len(self.fake.calls), 2)

    def test_cache_cannot_be_rebound_by_substituting_same_bytes_from_another_key(self):
        a = self.backend.build(request())
        first = next((self.root / 'cache').iterdir())
        self.backend.build(request(label='B'))
        second = next(path for path in (self.root / 'cache').iterdir() if path != first)
        (first / 'manifest.json').write_bytes((second / 'manifest.json').read_bytes())
        warm = self.backend.build(request(mode='incremental'))
        self.assertIsNotNone(a.cache_output_sha256)
        self.assertEqual(warm.status, 'executor_failed')
        self.assertEqual(len(self.fake.calls), 2)

    def test_real_source_contract_produces_full_build_request_and_verifies_id(self):
        from evolveengine.v3_level3_source import SourceContract

        contract = SourceContract(FILES, modes=request().modes, directories=('empty-dir',),
                                  provenance={'revision': SGLANG_REVISION, 'archive_sha256': SGLANG_ARCHIVE_SHA256})
        engine = contract.from_args({'files': {NATIVE_CPP: 'int other() { return 8; }'},
                                     'build_args': {'build_type': 'Debug', 'jobs': 8}})
        req = BuildRequest.from_contract(contract, engine, mode='cold')
        with self.assertRaises(ValueError):
            BuildRequest.from_contract(contract, replace(engine, digest='f' * 64))
        receipt = self.backend.build(req)
        invocation, tree, *_ = self.fake.calls[0]
        self.assertEqual(receipt.status, 'built')
        self.assertEqual(receipt.executor_kind, 'injected')
        self.assertEqual(receipt.candidate_id, engine.digest)
        self.assertEqual(receipt.effective_jobs, 2)
        self.assertEqual(json.loads(receipt.build_args_json), engine.build_args)
        self.assertEqual(tree, {**FILES, NATIVE_CPP: b'int other() { return 8; }'})
        self.assertIn('--config-settings=cmake.build-type=Debug', invocation.commands[0])
        native, engine_wheel = invocation.commands
        self.assertEqual((native[-1], engine_wheel[-1]), ('/work/src/' + AOT_ROOT, '/work/src/python'))
        self.assertIn('--config-settings=build-dir=/work/build/aot', native)
        self.assertIn('--config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS=1', native)
        self.assertIn('--config-settings=cmake.define.FETCHCONTENT_FULLY_DISCONNECTED=ON', native)
        for dep in ('cutlass', 'fmt', 'flashinfer'):
            self.assertIn(f'--config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_REPO-{dep.upper()}'
                          f'=/opt/ee-native/sources/{dep}', native)
        for command in (native, engine_wheel):
            self.assertEqual(command[:5], ('python3', '-I', '-m', 'pip', '--isolated'))
            self.assertTrue({'--no-index', '--no-build-isolation', '--no-deps'} <= set(command))

    def test_old_root_or_deleted_build_files_do_not_dispatch(self):
        files = {n: raw for n, raw in FILES.items() if n != AOT_ROOT + '/CMakeLists.txt'}
        with self.assertRaisesRegex(BuildError, 'pinned AOT'):
            self.backend.build(request(files={**files, 'sgl-kernel/CMakeLists.txt': b'wrong root'}))
        for args in ({'cuda_architectures': ['121a']}, {'cuda_fast_math': True},
                     {'cxx_optimization': '1'}, {'cuda_optimization': '0'}):
            with self.subTest(args=args), self.assertRaisesRegex(BuildError, 'does not support'):
                self.backend.build(request(build_args_json=json.dumps(args, separators=(',', ':')).encode('ascii')))
        self.assertFalse(self.fake.calls)

    def test_wheel_names_alone_cannot_qualify_native_artifacts(self):
        for data, reason in (
            (b'fake-wheel', 'invalid wheel archive'),
            (wheel_bytes(NATIVE_NAME, {k: v for k, v in NATIVE_MEMBERS.items() if k != SM100}), 'missing pinned AOT sm100'),
            (wheel_bytes(NATIVE_NAME, {**NATIVE_MEMBERS, SM100: ELF[:-2] + b'\x3e\x00'}), 'AArch64'),
            (wheel_bytes(NATIVE_NAME, {**NATIVE_MEMBERS, '../outside': b'unexpected'}), 'unsafe'),
        ):
            with self.subTest(reason=reason):
                self.fake.wheels[NATIVE_NAME] = data
                failed = self.backend.build(request())
                self.assertEqual(failed.status, 'invalid_output')
                self.assertIn(reason, failed.detail)
                self.assertFalse(failed.artifacts)
        self.fake.wheels = {NATIVE_NAME.replace('sglang_kernel', 'sgl_kernel'): WHEELS[NATIVE_NAME],
                            ENGINE_NAME: WHEELS[ENGINE_NAME]}
        self.assertEqual(self.backend.build(request()).status, 'invalid_output')
        self.assertFalse(list((self.root / 'artifacts').iterdir()))

    @unittest.skipUnless(os.environ.get('EVOLVEENGINE_NATIVE_ARCHIVE'), 'requires existing local pinned archive')
    def test_materialized_pinned_e0_matches_recipe_admission_without_compiling(self):
        from evolveengine.v3_level3_source import SourceContract

        repository = Path(__file__).resolve().parents[1]
        contract = SourceContract.acquire_spark26_e0(Path(os.environ['EVOLVEENGINE_NATIVE_ARCHIVE']), repository)
        self.assertEqual(len(contract.base), 12941)
        request = BuildRequest.from_contract(contract, contract.from_args({'files': {}}))
        self.assertFalse(any(p.startswith('sgl-kernel/') for p in request.files))
        self.assertIn(b'name = "sglang-kernel"', request.files[AOT_ROOT + '/pyproject.toml'])
        cmake = request.files[AOT_ROOT + '/CMakeLists.txt']
        for marker in (b'csrc/quantization/gguf/gguf_kernel.cu', b'code=sm_121a',
                       b'LIBRARY DESTINATION sgl_kernel/sm90', b'LIBRARY DESTINATION sgl_kernel/sm100',
                       b'DEFINED ENV{CCACHE_DIR}', b'FetchContent_Populate(repo-flashinfer)'):
            self.assertIn(marker, cmake)
        baseline = NativeBuildBackend(self.root, image=IMAGE, executor=self.fake)
        # Read-only preflight over the whole tree, not a fake report of compilation.
        source_hash = baseline._source_identity(request, time.monotonic() + 30)
        self.assertEqual(len(source_hash), 64)
        self.assertFalse(self.fake.calls)


class DockerPolicyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='ee-native-test-')
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        fake = FakeExecutor()
        backend = NativeBuildBackend(self.path / 'state', image=IMAGE, limits=SMALL, executor=fake)
        backend.build(request())
        self.job = replace(fake.calls[0][0], input_dir=self.path, bundle_path=self.path / 'spool')

    def test_no_privileged_inputs_or_shell_and_resource_limits(self):
        args = DockerExecutor.create_args(self.job)
        for flag in ('--pull=never', '--platform=linux/arm64', '--runtime=runc', '--network=none',
                     '--user=65532:65532', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                     '--read-only', '--no-healthcheck', '--restart=no', '--log-driver=none', '--ipc=private',
                     '--cpus=2', '--memory=256m', '--memory-swap=256m', '--pids-limit=512',
                     '--entrypoint=/usr/bin/env', 'NVIDIA_VISIBLE_DEVICES=void', 'CUDA_VISIBLE_DEVICES=',
                     'CARGO_NET_OFFLINE=true', 'SGLANG_BUILD_RUST_EXTS=all', 'CCACHE_DIR=/work/ccache'):
            self.assertIn(flag, args)
        self.assertEqual(args.count('--mount'), 1)
        self.assertEqual(args[args.index('--mount') + 1], f'type=bind,source={self.path},target=/ee-input,readonly')
        self.assertIn('size=32m', args[args.index('--tmpfs') + 1])
        self.assertEqual(args[args.index(IMAGE) + 1], '-i')
        self.assertNotIn('docker.sock', ' '.join(args))
        self.assertTrue(all(flags not in args for flags in ('--gpus', '--device', '--privileged', '--pid=host', '-v')))
        self.assertEqual(args[-1], '1')
        self.assertEqual(DockerExecutor.create_args(replace(self.job, mode='independent_cold'))[-1], '0')
        with self.assertRaisesRegex(BuildError, 'bind path'):
            DockerExecutor.create_args(replace(self.job, input_dir=self.path / ',src=/etc'))

    def test_mocked_daemon_exit_identity_and_cleanup_use_local_socket(self):
        completed = subprocess.CompletedProcess
        replies = [completed([], 0, b'linux/arm64 null\n'), completed([], 0, b'id\n'),
                   completed([], 0, b'exited 137 true\n'), completed([], 0, b'')]
        with patch('evolveengine.v3_level3_build.sys.platform', 'linux'), \
                patch('evolveengine.v3_level3_build.subprocess.run', side_effect=replies) as cli, \
                patch.object(DockerExecutor, '_attach', return_value=Execution(137, b'oom')) as attach:
            result = DockerExecutor().execute(self.job)
        self.assertEqual(result, Execution(137, b'oom', 'memory_limit'))
        attach.assert_called_once()
        self.assertEqual(cli.call_args_list[0].args[0][-4:], [
            'inspect', '--format', '{{.Os}}/{{.Architecture}} {{json (index .Config "Volumes")}}', IMAGE,
        ])
        self.assertEqual(cli.call_args_list[-1].args[0][-3:], ['rm', '-f', self.job.job_id])
        self.assertTrue(all('unix:///var/run/docker.sock' in call.args[0] for call in cli.call_args_list))
        self.assertTrue(all(set(call.kwargs['env']) == {'PATH', 'HOME', 'LANG'} for call in cli.call_args_list))

    def test_declared_image_volumes_rejected_before_create(self):
        reply = subprocess.CompletedProcess([], 0, b'linux/arm64 {"/work":{}}\n')
        with patch('evolveengine.v3_level3_build.sys.platform', 'linux'), \
                patch('evolveengine.v3_level3_build.subprocess.run', return_value=reply) as cli:
            with self.assertRaisesRegex(BuildError, 'declares volumes'):
                DockerExecutor().execute(self.job)
        cli.assert_called_once()

    def test_mocked_failed_removal_overrides_success(self):
        completed = subprocess.CompletedProcess
        replies = [completed([], 0, b'linux/arm64 null\n'), completed([], 0, b'id\n'),
                   completed([], 0, b'exited 0 false\n'), completed([], 1, b'')]
        with patch('evolveengine.v3_level3_build.sys.platform', 'linux'), \
                patch('evolveengine.v3_level3_build.subprocess.run', side_effect=replies), \
                patch.object(DockerExecutor, '_attach', return_value=Execution(0)):
            with self.assertRaises(CleanupUncertain):
                DockerExecutor().execute(self.job)

    def test_attach_pipe_caps_and_deadline_with_local_fixture_process(self):
        # Exercise the real pipe-spooling loop with a Python fixture, not Docker or a build script.
        launch = subprocess.Popen

        def attach(code, *, limit=5, **kwargs):
            invocation = replace(self.job, deadline=time.monotonic() + limit,
                                 bundle_path=self.path / ('spool-' + str(kwargs.get('name', 'a'))),
                                 limits=replace(self.job.limits, files=1, artifact_bytes=32, cache_bytes=32))
            def local_child(unused, **options):
                options.pop('env')
                return launch([sys.executable, '-c', code], **options)
            with patch('evolveengine.v3_level3_build.subprocess.Popen', side_effect=local_child):
                result = DockerExecutor._attach(['not-docker'], dict(os.environ), self.path, invocation)
            return result, invocation

        result, job = attach("import os; os.write(1,b'ok'); os.write(2,b'x'*30000+b'end')")
        self.assertEqual((result.status, result.exit_code), ('exited', 0))
        self.assertEqual(job.bundle_path.read_bytes(), b'ok')
        self.assertEqual(result.log_tail, b'x' * 13 + b'end')
        result, job = attach("import os; os.write(1,b'x'*100000)", name='overflow')
        self.assertEqual(result.status, 'output_limit')
        self.assertLessEqual(job.bundle_path.stat().st_size, job.limits.stream_bytes)
        result, _ = attach('import time; time.sleep(10)', name='timeout', limit=0.25)
        self.assertEqual(result.status, 'timeout')


if __name__ == '__main__':
    unittest.main()
