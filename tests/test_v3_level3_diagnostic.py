from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from evolveengine import v3_level3_diagnostic as d


class DiagnosticTests(TestCase):
    def test_no_eval_on_host_and_bad_payloads(self):
        valid = 'raise RuntimeError("must not be run to validate syntax")'
        self.assertEqual(d.request(valid),valid.encode())
        for src in (1,'','\x00', 'foo('*2, ' '* (d.MAX_SCRIPT+1)):
            with self.subTest(src=type(src)), self.assertRaises(ValueError):
                d.request(src)

    def test_check_exact_gpu_half_private_runtime_and_two_read_only_mounts(self):
        a, b, g, n = Path('/readonly/dep'),Path('/readonly/diagnostic'),'GPU-abcdef','ee-l3-job-diag-x'
        image = 'sha256:'+'e'*64
        dep = SimpleNamespace(tree=a)
        view = dict(Name='/'+n, Image=image, Id='a'*64, State={'Running':False},
                    HostConfig=dict(NetworkMode='none',ReadonlyRootfs=True,Privileged=False,PidMode='',IpcMode='private',
                                    PortBindings={},Devices=[],CapDrop=['ALL'],CapAdd=[],SecurityOpt=['no-new-privileges'],
                                    PidsLimit=512,Memory=d.MEMORY_MIB*(1<<20),Tmpfs=d.runtime_tmpfs(d.SCRATCH_MIB),
                                    DeviceRequests=[dict(DeviceIDs=[g],Count=0,Capabilities=[['gpu']])]),
                    Config=dict(User='12345:12345',Entrypoint=[d.ENTRY],Cmd=list(d.ARGV),
                                Env=[f'{k}={v}' for k,v in d.RUNTIME_ENV.items()]),
                    Mounts=[dict(Destination=dest,Source=str(src),RW=False,Type='bind')
                            for dest,src in ((d.TARGET,a),(d.MOUNT,b))])
        verify=lambda: d.attest(view,dep,image,g,b,n)
        self.assertEqual(verify(),'a'*64)
        view['Mounts'].append(dict(Destination='/model',Source='/private/model',RW=False,Type='bind'))
        with self.assertRaisesRegex(ValueError,'scope drift'):
            verify()
        view['Mounts'].pop()
        view['HostConfig']['NetworkMode']='default'
        with self.assertRaises(ValueError):
            verify()

    def test_no_docker_for_bad_budget(self):
        with TemporaryDirectory() as folder, patch.object(d.DockerDiagnostic,'_call') as invocation:
            runner=d.DockerDiagnostic(None,docker='/usr/bin/docker',exclusive=None)
            with self.assertRaisesRegex(ValueError,'lease outside'):
                runner.launch(None,'e'*64,'pass',Path(folder),d.MAX_MS+1)
            invocation.assert_not_called()
