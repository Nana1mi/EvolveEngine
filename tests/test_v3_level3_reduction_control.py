from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine.v3_harness import digest
from evolveengine.v3_level3_build import BuildError
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import Deployment, RuntimeFile
from experiments.spark26_gguf.level3.reduction_control import archived


class ArchivedReductionControl(TestCase):
    def fixture(self, root):
        compound='a'*64
        job=root/'worker/job-native'; folder=job/'deployment'
        site=folder/'site'; site.mkdir(parents=True); site.chmod(0o755)
        binary=b'test native payload'; a=site/'fake.so'; a.write_bytes(binary); a.chmod(0o444)
        deploy=Deployment(compound,job.name,(('test.whl','c'*64),),
            (RuntimeFile('fake.so',len(binary),digest(binary)),),folder,'d'*64)
        write(folder/'manifest.json',deploy.record())
        comp=write(job/'component-build.json',dict(status='built',candidate=compound,deployment=deploy.identity))
        data=dict(deployment=deploy.identity,component_receipt=comp,build_mode='component_native',
                  base_wheels=dict(deploy.wheels),parent_deployment=deploy.parent_identity)
        b=dict(engine=compound,status='built',data=data)
        evidence=write(job/'result.json',dict(b,deployment=deploy.identity,observation_pin=None,observation_unavailable=False))
        pub=write(root/'candidate.json',dict(verified=True,result=dict(b,evidence=evidence)))
        bad=write(root/'quality.json',dict(verified=True,result=dict(engine=compound,status='violation',build=evidence,
                  data=dict(trajectory_screen=dict(failed=['iso_eos_logp','iso_probability'])))))
        return compound,pub,bad,deploy,a,binary

    def test_verifies_pinned_build_and_bytes_before_using_archive(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); compound,pub,bad,expected,a,binary=self.fixture(root)
            try:
                deployment,row=archived(root,pub,bad,compound)
                self.assertEqual(deployment,expected)
                self.assertEqual(deployment.runtime_config,b'')
                self.assertEqual(row['data']['deployment'],expected.identity)
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    archived(root,pub,'0'*64,compound)
                a.chmod(0o644); a.write_bytes(binary[::-1]); a.chmod(0o444)
                with self.assertRaisesRegex(BuildError,'deployment file changed'):
                    archived(root,pub,bad,compound)
            finally:
                a.chmod(0o644)

    def test_rejects_duplicate_or_unbound_raw_build(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); compound,pub,bad,_,a,_=self.fixture(root)
            try:
                existing=root/'worker/job-native/result.json'
                other=root/'worker/job-unbound/result.json'; other.parent.mkdir()
                other.write_bytes(existing.read_bytes())
                with self.assertRaisesRegex(ValueError,'unique'):
                    archived(root,pub,bad,compound)
            finally:
                a.chmod(0o644)
