import os
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic
from unittest import TestCase, skipUnless

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_build import BuildError, CleanupUncertain, NativeBuildBackend
from evolveengine.v3_level3_diagnostic import DiagnosticResult
from evolveengine.v3_level3_deploy import (DockerService, IMMEDIATE_GRAPH_WIRE, JIT_CACHE, LAUNCH, RUNTIME_ENV, ServiceSpec, TARGET, WEIGHT,
                                           launch_args, runtime_container_args, runtime_tmpfs, stage_deployment, validate_container)
from evolveengine.v3_level3_source import SourceContract, SGLANG_REVISION, SGLANG_ARCHIVE_SHA256
from evolveengine.v3_level3_worker import NativeWorker, ObservationUnavailable
from test_v3_level3_build import (ENGINE_MEMBERS, ENGINE_NAME, FakeExecutor, FILES, IMAGE,
                                 SMALL, SM100, request, wheel_bytes)

GPU = 'GPU-00000000-0000-0000-0000-000000000001'


class DeployTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.executor = FakeExecutor()
        self.native = NativeBuildBackend(self.root/'build',image=IMAGE,limits=SMALL,executor=self.executor)

    def test_incremental_wire_keeps_both_graphs_and_wire_interval_locked(self):
        from dataclasses import replace
        base = ServiceSpec(IMAGE,IMAGE,self.root/'model',(),self.root/'weight','a'*64,GPU,'graph_defaults')
        spec = replace(base, launch_profile=IMMEDIATE_GRAPH_WIRE)
        old = launch_args(base)
        actual = launch_args(spec)
        self.assertEqual(actual, (*old, '--incremental-streaming-output'))
        self.assertEqual(actual[actual.index('--stream-interval')+1], '1')
        self.assertNotIn('--disable-decode-cuda-graph', actual)
        self.assertNotIn('--disable-prefill-cuda-graph', actual)
        deploy = stage_deployment(self.native.build(request()),self.root/'deploy')
        candidate = replace(deploy, runtime_config=b'{"radix-cache":true}')
        self.assertEqual(launch_args(spec,candidate), tuple(a for a in actual if a != '--disable-radix-cache'))
        with self.assertRaises(ValueError):
            launch_args(spec, replace(deploy,runtime_config=b'{"incremental-streaming-output":false}'))

    def test_separately_bound_gguf_requires_an_empty_regular_mount_point(self):
        model = self.root/'model'
        model.mkdir()
        assets = {'tokenizer.json':b'{}','chat_template.jinja':b'fixture'}
        pins = tuple((name,sha256(value).hexdigest()) for name,value in assets.items())
        for name,value in assets.items():
            (model/name).write_bytes(value)
        placeholder = model/WEIGHT
        placeholder.touch()
        gguf = self.root/'actual.gguf'
        gguf.write_bytes(b'separately pinned GGUF fixture')
        spec = ServiceSpec(IMAGE,IMAGE,model,pins,gguf,sha256(gguf.read_bytes()).hexdigest(),GPU)
        self.assertEqual(spec.recheck_model(deadline=monotonic()+5)[WEIGHT],spec.gguf_sha256)
        placeholder.write_bytes(b'not an empty mount point')
        with self.assertRaisesRegex(BuildError,'invalid GGUF bind mount point'):
            spec.recheck_model(deadline=monotonic()+5)

    def test_wheels_bound_to_runtime_but_not_imported(self):
        built = self.native.build(request())
        deploy = stage_deployment(built,self.root/'deploy').verify()
        self.assertEqual(deploy.candidate,built.candidate_id)
        self.assertEqual(dict(deploy.wheels),built.artifact_hashes)
        self.assertEqual(deploy.native_mapping('sm100')[0],{SM100:f'{TARGET}/{SM100}'})
        self.assertFalse((self.root/'deploy-site-executed').exists())
        path = deploy.tree/'sglang/__init__.py'
        path.chmod(0o644)
        path.write_bytes(b'!' * path.stat().st_size)
        with self.assertRaisesRegex(BuildError,'changed'):
            deploy.verify()

    def test_unsafe_or_excess_wheels_cannot_commit(self):
        for leaf in ('sitecustomize.py','sglang/ignored.pth','sglang/__pycache__/bad.pyc'):
            with self.subTest(leaf=leaf):
                self.executor.wheels[ENGINE_NAME] = wheel_bytes(ENGINE_NAME,{**ENGINE_MEMBERS,leaf:b'test'})
                built = self.native.build(request())
                self.assertEqual(built.status,'built')
                dest = self.root/'rejected'
                with self.assertRaises(BuildError):
                    stage_deployment(built,dest)
                self.assertFalse(dest.exists())
        self.executor = FakeExecutor()
        self.native = NativeBuildBackend(self.root/'build-2',image=IMAGE,limits=SMALL,executor=self.executor)
        with self.assertRaisesRegex(BuildError,'expanded wheel budget'):
            stage_deployment(self.native.build(request()),self.root/'over',max_expanded=8)

    @skipUnless(os.environ.get('EVOLVEENGINE_TEST_DOCKER_IMAGE'),'requires a local pinned Python Docker image')
    def test_real_docker_imports_only_bound_offline_packages(self):
        deploy = stage_deployment(self.native.build(request()),self.root/'deploy')
        image = os.environ['EVOLVEENGINE_TEST_DOCKER_IMAGE']
        args = runtime_container_args(deploy,image=image,name='ee-l3-source-import-test',scratch_mib=32,memory_mib=256)
        code = '''import pathlib, sglang, sgl_kernel
for mod in (sglang,sgl_kernel): assert mod.__file__.startswith('/ee-candidate/')
try: pathlib.Path(sglang.__file__).write_text('tamper')
except OSError: pass
else: raise AssertionError('candidate mount was writable')
assert pathlib.Path('/sys/class/net').is_dir()
assert sorted(x.name for x in pathlib.Path('/sys/class/net').iterdir())==['lo']
assert pathlib.Path('/tmp/write-check').write_text('scratch')==7
print('read-only-candidate-loaded')'''
        docker = shutil.which('docker')
        run = subprocess.run([docker,'run','--rm',*args,'--entrypoint','python',image,'-c',code],
                             capture_output=True,timeout=35)
        self.assertEqual(run.returncode,0,run.stderr.decode(errors='replace'))
        self.assertEqual(run.stdout.strip(),b'read-only-candidate-loaded')
        deploy.verify()

    def test_native_worker_build_receipt_revalidates_staged_bytes(self):
        contract = SourceContract(FILES,provenance=dict(revision=SGLANG_REVISION,archive_sha256=SGLANG_ARCHIVE_SHA256))
        engine = contract.from_args({'files':{}})
        @contextmanager
        def exclusive(deadline):
            self.assertGreater(deadline,monotonic())
            yield
        class NoService:
            def start(*_):
                raise AssertionError('build must not start a GPU service')
        worker = NativeWorker(self.root/'worker',contract,self.native,NoService(),exclusive=exclusive,collect=None)
        build = worker('develop_build',engine,None,20_000)
        self.assertTrue(worker.verify(build))
        self.assertEqual(build.status,'built')
        path = worker.deployments[build.evidence].tree/SM100
        path.chmod(0o644)
        path.write_bytes(b'not the observed mapping')
        self.assertFalse(worker.verify(build))

    def test_agent_diagnostic_is_bound_to_a_verified_build_and_returned_output(self):
        contract = SourceContract(FILES,provenance=dict(revision=SGLANG_REVISION,archive_sha256=SGLANG_ARCHIVE_SHA256))
        engine = contract.from_args({'files':{}})
        @contextmanager
        def exclusive(deadline):
            self.assertGreater(deadline,monotonic())
            yield
        class NoService:
            def start(*_):
                raise AssertionError('diagnostic uses its own isolated runner')
        class Diagnostic:
            def launch(_self,deployment,build,source,root,cap):
                archive = root/'diag-fixture'
                archive.mkdir()
                output = b'diagnostic-output\n'
                (archive/'observer.stdout').write_bytes(output)
                script = digest(source.encode())
                inner = dict(engine=deployment.candidate,build=build,script=script,
                             status='completed_non_scoring',stdout_bytes=len(output),
                             stdout_sha256=digest(output),verified_removed=True,non_scoring=True)
                raw = canonical(inner)+b'\n'
                (archive/'result.json').write_bytes(raw)
                result = DiagnosticResult(deployment.candidate,build,script,'completed_non_scoring',9,
                                          digest(raw),len(output),digest(output),None)
                return result,archive
        worker = NativeWorker(self.root/'worker',contract,self.native,NoService(),exclusive=exclusive,
                              collect=None,diagnostic=Diagnostic())
        build = worker('develop_build',engine,None,20_000)
        result = worker.diagnose(engine,build,"print('probe')\n",20_000)
        self.assertTrue(worker.verify(result))
        self.assertEqual(result.status,'completed_non_scoring')
        self.assertEqual(result.data['stdout'],'diagnostic-output\n')
        self.assertTrue(result.data['non_scoring'])
        result.data['stdout'] = 'forged'
        self.assertFalse(worker.verify(result))

    def test_gpu_container_mount_and_device_drift_denied(self):
        deploy = stage_deployment(self.native.build(request()),self.root/'deploy')
        spec = ServiceSpec(IMAGE,'sha256:'+'a'*64,self.root/'model',(),self.root/'weight','a'*64,GPU)
        raw = dict(Name='/ee-l3-test',Image=spec.image_id,State=dict(Running=False),Id='b'*64,
                   HostConfig=dict(NetworkMode='none',ReadonlyRootfs=True,Privileged=False,PidMode='',IpcMode='private',
                                   PortBindings={},Devices=[],CapDrop=['ALL'],CapAdd=[],SecurityOpt=['no-new-privileges'],
                                   Tmpfs=runtime_tmpfs(),PidsLimit=512,Memory=65536*1024**2,
                                   DeviceRequests=[dict(DeviceIDs=[GPU],Count=0,Capabilities=[['gpu']])]),
                   Config=dict(User='12345:12345',Entrypoint=['python'],Cmd=list(LAUNCH),Env=[f'{k}={v}' for k,v in RUNTIME_ENV.items()]),
                   Mounts=[dict(Destination=dest,Source=str(src),RW=False,Type='bind') for dest,src in
                           ((TARGET,deploy.tree),('/model',spec.model_dir),(f'/model/{WEIGHT}',spec.gguf))])
        self.assertEqual(validate_container(raw,deploy,spec,'ee-l3-test'),'b'*64)
        self.assertIn('noexec',raw['HostConfig']['Tmpfs']['/tmp'].split(','))
        self.assertIn('exec',raw['HostConfig']['Tmpfs'][JIT_CACHE].split(','))
        jit = f'SGLANG_JIT_CACHE_DIR={JIT_CACHE}/sglang-cache'
        raw['Config']['Env'].remove(jit)
        with self.assertRaises(BuildError):
            validate_container(raw,deploy,spec,'ee-l3-test')
        raw['Config']['Env'].append(jit)
        raw['HostConfig']['Tmpfs'][JIT_CACHE]=raw['HostConfig']['Tmpfs'][JIT_CACHE].replace(',exec,',',noexec,')
        with self.assertRaisesRegex(BuildError,'pinned runtime'):
            validate_container(raw,deploy,spec,'ee-l3-test')
        raw['HostConfig']['Tmpfs']=runtime_tmpfs()
        raw['Mounts'][0]['RW']=True
        with self.assertRaisesRegex(BuildError,'pinned runtime'):
            validate_container(raw,deploy,spec,'ee-l3-test')
        raw['Mounts'][0]['RW']=False
        raw['HostConfig']['DeviceRequests'][0]['DeviceIDs']=[]
        with self.assertRaisesRegex(BuildError,'pinned runtime'):
            validate_container(raw,deploy,spec,'ee-l3-test')

    def test_repeat_build_and_clean_observer_failure(self):
        contract = SourceContract(FILES,provenance=dict(revision=SGLANG_REVISION,archive_sha256=SGLANG_ARCHIVE_SHA256))
        candidate = contract.from_args({'files':{}})
        order = []
        @contextmanager
        def exclusive(deadline):
            try:
                yield
            finally:
                order.append('exclusive-cleared')
        class Service:
            model_hashes = {}
            @contextmanager
            def start(self,*args):
                try:
                    yield self
                finally:
                    order.append('service-cleared')
        def collection(*_):
            order.append('collector-stopped')
            raise ObservationUnavailable('fixture')
        worker = NativeWorker(self.root/'worker',contract,self.native,Service(),exclusive=exclusive,collect=collection)
        cold = worker('develop_build',candidate,None,20000)
        cache = worker('develop_build',candidate,cold,20000)
        self.assertEqual((cold.data['build_mode'],cache.data['build_mode'],cache.build),('cold','incremental',None))
        self.assertNotEqual(cold.evidence,cache.evidence)
        self.assertTrue(worker.verify(cold) and worker.verify(cache))
        q = worker('develop_quality',candidate,cache,20000)
        self.assertEqual(q.build,cache.evidence)
        self.assertEqual(q.status,'insufficient_evidence')
        self.assertTrue(worker.verify(q))
        self.assertEqual(order[-3:],['collector-stopped','service-cleared','exclusive-cleared'])
        self.assertFalse(worker.quarantined)

        @contextmanager
        def dirty(deadline):
            yield
            raise CleanupUncertain('fixture')
        worker.exclusive = dirty
        with self.assertRaises(CleanupUncertain):
            worker('develop_quality',candidate,cache,20000)
        self.assertTrue(worker.quarantined)
