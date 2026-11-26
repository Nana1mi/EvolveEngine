from contextlib import nullcontext
from dataclasses import replace
import json
import os
import stat
from unittest import skipUnless
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine.v3_level3_build import AOT_GGUF, ENGINE_GGUF, Execution, NativeBuildBackend
from evolveengine.v3_level3_components import _compile_parallelism
from evolveengine.v3_level3_deploy import MAPS
from evolveengine.v3_level3_source import SGLANG_ARCHIVE_SHA256, SGLANG_REVISION, SourceContract
from evolveengine.v3_level3_worker import NativeWorker
from test_v3_level3_build import ELF, ENGINE_MEMBERS, ENGINE_NAME, FILES, IMAGE, SMALL, FakeExecutor, wheel_bytes


class DevelopmentExecutor(FakeExecutor):
    def __init__(self):
        super().__init__()
        members = {**ENGINE_MEMBERS, ENGINE_GGUF.removeprefix('python/'): FILES[ENGINE_GGUF]}
        members['sglang/srt/rust_extensions/_test.cpython-312-aarch64-linux-gnu.so'] = ELF
        self.wheels[ENGINE_NAME] = wheel_bytes(ENGINE_NAME, members)
        self.plans = []
        self.jobs = []
        self.mount_modes = []
        self.emit_extra = False

    def execute(self, job):
        plan_path = job.input_dir/'plan.json'
        if not plan_path.exists():
            return super().execute(job)
        self.mount_modes.append(([stat.S_IMODE((job.input_dir/name).stat().st_mode) for name in
                                  ('driver.py','plan.json','tree-plan.json')],
                                 stat.S_IMODE(job.input_dir.stat().st_mode),
                                 stat.S_IMODE(job.input_dir.parent.stat().st_mode)))
        plan = json.loads(plan_path.read_bytes())
        cache = [(p.relative_to(job.input_dir/'cache').as_posix(), p.read_bytes())
                 for p in (job.input_dir/'cache').rglob('*') if p.is_file()]
        self.plans.append((plan, cache))
        self.jobs.append(job.jobs)
        with job.bundle_path.open('xb') as spool:
            spool.write(b'level3-component-v1\n')
            def emit(lane, name, data):
                spool.write(json.dumps([lane,name,len(data)]).encode()+b'\n'+data)
            if 'native' in plan:
                for target in MAPS:
                    emit('module', target, ELF+b'new native')
                emit('cache','a/new',b'cache for child')
            if 'rust' in plan:
                emit('module','sglang/srt/rust_extensions/_test.cpython-312-aarch64-linux-gnu.so',ELF+b'new rust')
            if self.emit_extra:
                emit('module','not-allowed.so',ELF)
            times = dict.fromkeys(['prepare_source_and_dependencies', *plan, 'export_modules_and_cache'], 1)
            spool.write(json.dumps(['durations',times]).encode()+b'\n')
        return Execution(0)


class ComponentTests(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.exe = DevelopmentExecutor()
        self.limits = replace(SMALL, cpus=6, memory_mib=81920)
        native = NativeBuildBackend(self.root/'store',image=IMAGE,executor=self.exe,limits=self.limits)
        self.contract = SourceContract(FILES,provenance=dict(revision=SGLANG_REVISION,archive_sha256=SGLANG_ARCHIVE_SHA256))
        self.worker = NativeWorker(self.root/'worker',self.contract,native,None,exclusive=lambda _:nullcontext(),collect=None)
        self.worker.component_builds = True
        self.e0 = self.contract.from_args({'files': {}})
        self.base = self.worker('develop_build', self.e0, None, 10000)

    def build(self, edits, parent=None):
        engine = self.contract.from_args({'files': edits})
        return self.worker('develop_build',engine,parent or self.base,10000)

    def test_python_and_triton_overlay_without_compiler_or_wheel(self):
        name = 'python/sglang/srt/layers/quantization/new_triton_kernel.py'
        child = self.build({ENGINE_GGUF:'changed\n',name:'import triton\n'})
        self.assertEqual(child.data['build_mode'], 'component_python')
        self.assertEqual(len(self.exe.calls),1)
        self.assertFalse(self.exe.plans)
        deployed = self.worker.deployments[child.evidence]
        self.assertEqual((deployed.tree/name[7:]).read_bytes(),b'import triton\n')
        self.assertEqual(child.data['base_wheels'],self.base.data['wheels'])
        self.assertTrue(self.worker.verify(child))
        self.assertTrue(self.worker.verify(self.base))
        restored = self.build({},child)
        original = self.worker.deployments[self.base.evidence]
        self.assertEqual(self.worker.deployments[restored.evidence].files,original.files)
        self.assertTrue(self.worker.verify(child))

    def test_json_tuning_config_is_deployed_replayable_and_removable(self):
        name = 'python/sglang/srt/layers/quantization/configs/gb10.json'
        child = self.build({name: '{"num_warps":4}'})
        self.assertEqual(child.status, 'built')
        self.assertTrue(self.worker.verify(child))
        deployed = self.worker.deployments[child.evidence]
        self.assertEqual((deployed.tree/name[7:]).read_bytes(), b'{"num_warps":4}')
        reverted = self.build({}, child)
        self.assertFalse((self.worker.deployments[reverted.evidence].tree/name[7:]).exists())
        self.assertTrue(self.worker.verify(reverted))

    @skipUnless(os.name=='posix','POSIX UID mount permissions')
    def test_private_worker_umask_exposes_only_readonly_component_metadata(self):
        old=os.umask(0o077)
        try: result=self.build({AOT_GGUF:'fixture CUDA change'})
        finally: os.umask(old)
        self.assertEqual(result.status,'built')
        self.assertEqual(self.exe.mount_modes,[([0o444]*3,0o755,0o700)])

    @skipUnless(os.name=='posix','POSIX UID mount permissions')
    def test_private_worker_umask_exposes_the_mounted_site_but_not_host_state(self):
        from stat import S_IMODE
        from evolveengine.v3_level3_build import BuildError
        from evolveengine.v3_level3_deploy import runtime_container_args
        old=os.umask(0o077)
        try:
            result=self.build({ENGINE_GGUF:'fixture Python change'})
        finally:
            os.umask(old)
        self.assertEqual(result.status,'built')
        child=self.worker.deployments[result.evidence]
        parent=self.worker.deployments[self.base.evidence]
        mode=lambda path:S_IMODE(path.stat().st_mode)
        self.assertEqual((mode(child.root.parent),mode(child.root),mode(child.root/'manifest.json')),(0o700,0o700,0o600))
        for current,_,names in os.walk(child.tree):
            folder=Path(current)
            self.assertEqual(mode(folder),0o755)
            self.assertTrue(all(mode(folder/name)==0o444 for name in names))
        self.assertEqual(mode(parent.tree/'sglang/__init__.py'),0o444)
        self.assertIs(child.verify_runtime_access(),child)
        child.tree.joinpath('sglang').chmod(0o700)
        # File bytes can still verify as owner; the non-root container must not start.
        self.assertIs(child.verify(),child)
        with self.assertRaisesRegex(BuildError,'not traversable'):
            runtime_container_args(child,image=IMAGE,name='ee-l3-blocked-bind')
        self.assertEqual(mode(parent.tree/'sglang'),0o755)

    def test_native_builder_ram_tier_reaches_invocation_and_receipt(self):
        limits = self.limits
        self.assertEqual(_compile_parallelism(1, limits.memory_mib, True), (1, 1))
        self.assertEqual(_compile_parallelism(6, 1024, True), (1, 1))
        self.assertEqual(_compile_parallelism(6, limits.memory_mib, False), (6, 1))
        child = self.build({AOT_GGUF: 'fixture CUDA change'})
        self.assertEqual(child.status, 'built')
        self.assertEqual(self.exe.jobs, [2])
        self.assertIn('-DSGL_KERNEL_COMPILE_THREADS=1', self.exe.plans[0][0]['configure'])
        self.assertEqual((child.data['compile_jobs'], child.data['nvcc_threads']), (2, 1))

    def test_native_delta_uses_parent_cache_and_replaces_only_bound_modules(self):
        first = self.build({AOT_GGUF:'first CUDA change'})
        second = self.build({AOT_GGUF:'second CUDA change',ENGINE_GGUF:'new dispatch'},first)
        self.assertEqual([list(plan) for plan,_ in self.exe.plans],[['configure','native']]*2)
        self.assertEqual(dict(self.exe.plans[0][1]),self.exe.cache)
        self.assertEqual(dict(self.exe.plans[1][1]),{'a/new':b'cache for child'})
        self.assertEqual(first.data['cache_out'],second.data['cache_in'])
        self.assertEqual(second.data['parent_build'],first.evidence)
        self.assertIn('native',second.data['container_reported_phase_ms'])
        original = self.worker.deployments[self.base.evidence]
        now = self.worker.deployments[second.evidence]
        for name in MAPS:
            self.assertEqual((now.tree/name).read_bytes(), ELF+b'new native')
            self.assertEqual((original.tree/name).read_bytes(), ELF)
        self.assertEqual((now.tree/ENGINE_GGUF[7:]).read_bytes(), b'new dispatch')
        self.assertTrue(self.worker.verify(second))

    def test_rust_build_skips_native_and_python_packaging(self):
        child = self.build({'rust/sglang-server/src/lib.rs':'mod internal;'})
        plan, = [p for p,_ in self.exe.plans]
        self.assertEqual(list(plan), ['rust'])
        self.assertIn('build_ext',plan['rust'])
        self.assertEqual(child.status,'built')
        self.assertTrue(self.worker.verify(child))

    def test_build_inputs_and_modified_parent_recipes_use_full_cold_path(self):
        child = self.build({'python/setup.py':'# changed build hook'})
        descendant = self.build({'python/setup.py':'# changed build hook',ENGINE_GGUF:'changed'},child)
        self.assertEqual([r.data['build_mode'] for r in (child, descendant)],['cold','cold'])
        self.assertEqual(len(self.exe.calls),3)
        self.assertFalse(self.exe.plans)

    def test_unexpected_module_fails_job_without_altering_parent(self):
        self.exe.emit_extra = True
        child = self.build({AOT_GGUF:'changed'})
        self.assertEqual(child.status,'invalid_output')
        self.assertTrue(self.worker.verify(child))
        self.assertNotIn(child.evidence,self.worker.deployments)
        self.assertTrue(self.worker.verify(self.base))

    def test_final_independent_build_bypasses_components_and_does_not_publish_a_cache(self):
        child = self.contract.from_args({'files':{AOT_GGUF:'changed'}})
        self.assertEqual(self.build({AOT_GGUF:'changed'}).data['build_mode'],'component_native')
        result = self.worker.independent_build(child,10000)
        self.assertEqual(result.data['build_mode'],'independent_cold')
        self.assertIsNone(result.build)
        self.assertTrue(self.worker.verify(result))
        native = self.worker.records[result.evidence][2]
        self.assertIsNone(native.cache_input_sha256)
        self.assertIsNone(native.cache_output_sha256)
        self.assertEqual(len(self.exe.plans),1)
        self.assertEqual(self.exe.calls[-1][2],{})
