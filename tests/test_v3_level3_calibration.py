from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine.v3_harness import canonical, digest
from experiments.spark26_gguf.level3.calibrate_nvml_pid import MIB, PLAN, evaluate, verify_calibration


class CalibrationTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.node = dict(worker_id='spark-c3bc',boot_id='boot',device_uuid='GPU-one',driver='driver',compute_pids=[])
        self.rows = []
        for trial in PLAN['order']:
            cpu = trial.startswith('cpu-')
            level = 32*MIB if not cpu else None
            held = level+int(trial.split('-')[1])*MIB if not cpu else None
            def sample(value):
                return [dict(pid=101,used_bytes=value,raw='' if value is None else f'GPU-one, 101, {value//MIB}\n')]*3
            self.rows.append(dict(trial=trial,before=sample(level),held=sample(held),after=sample(level)))

    def save(self, name, obj):
        b = canonical(obj)+b'\n'
        (self.root/name).write_bytes(b)
        return digest(b)

    def stage(self):
        plan_pin = self.save('plan.json',PLAN)
        self.save('node.json',self.node)
        raw = b''.join(canonical(r)+b'\n' for r in self.rows)
        (self.root/'samples.jsonl').write_bytes(raw)
        return self.save('result.json',dict(**evaluate(self.rows),node=self.node,plan_sha256=plan_pin,raw_sha256=digest(raw)))

    def test_exact_report_replay_and_boot_binding(self):
        pin = self.stage()
        got = verify_calibration(self.root,pin,self.node)
        self.assertEqual(got['smallest_calibrated_gpu_delta_bytes'],12*MIB)
        self.assertEqual(got['role'],'engineering_E0_observation_not_candidate_residency_gate')
        for key in ('worker_id','boot_id','device_uuid','driver'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'another node'):
                verify_calibration(self.root,pin,{**self.node,key:'other'})

    def test_incomplete_or_tampered_raw_observations_do_not_pass(self):
        pin = self.stage()
        sample = self.root/'samples.jsonl'
        sample.write_bytes(sample.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'raw hash mismatch'):
            verify_calibration(self.root,pin,self.node)
        self.rows[1]['held'][0]['raw'] = 'GPU-one, 101, [N/A]\n'
        with self.assertRaisesRegex(ValueError,'unavailable'):
            verify_calibration(self.root,self.stage(),self.node)
