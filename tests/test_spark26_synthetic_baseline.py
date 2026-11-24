import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'synthetic_baseline_audit', Path(__file__).resolve().parents[1] /
    'experiments/spark26_gguf/synthetic_baseline_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SyntheticBaselineTests(unittest.TestCase):
    def test_one_prefill_required_for_atomic_pair(self):
        lock, _ = audit.load_lock(audit.LOCK)
        header = 'EE_CALIBRATION long2 timed wave-0\n'
        footer = 'EE_CALIBRATION complete\n'
        combined = 'Prefill batch, #new-seq: 2, #new-token: 1536, #cached-token: 0,\n'
        split = 'Prefill batch, #new-seq: 1, #new-token: 768, #cached-token: 0,\n'
        audit.check_prefills(header+combined+footer, lock, [('long2','timed',0)])
        for lines in (split+split, combined.replace('cached-token: 0','cached-token: 1'), ''):
            with self.assertRaises(ValueError):
                audit.check_prefills(header+lines+footer, lock, [('long2','timed',0)])

    def test_full_evidence_and_missing_record(self):
        lock, digest = audit.load_lock(audit.LOCK)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pair in lock['pairs']:
                for label in pair['labels']:
                    rows = [dict(kind='run_identity', label=label, workload_sha256=digest)]
                    logs = []
                    for name, phase, i in audit.schedule(lock, pair['shape_order']):
                        n, w, out = (lock['shapes'][name][x] for x in ('input_tokens','width','output_tokens'))
                        logs += [f'EE_CALIBRATION {name} {phase} wave-{i}',
                                 f'Prefill batch, #new-seq: {w}, #new-token: {w*n}, #cached-token: 0,']
                        rows.append(dict(kind='wave', shape=name, index=i, phase=phase, wall_seconds=1,
                            requests=[dict(rid=f'{label}-{name}-{phase}-{i}-{j}', input_tokens=n,
                                           first_token_ms=80, finish='length', output_sha256='0'*64,
                                           output_tokens=out) for j in range(w)]))
                    (root/f'{label}.stdout').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
                    (root/f'{label}.server.log').write_text('\n'.join(logs+['EE_CALIBRATION complete']), encoding='utf-8')
            result = audit.summarize(root, lock, digest)
            self.assertEqual(result['n_pairs'], 3)
            self.assertEqual(result['timed_requests'], 180)
            self.assertEqual(result['pairs'][1]['run_order'], ['b2', 'a2'])
            self.assertEqual(result['max_abs_pct']['short2']['wave_wall_ms'], 0)
            self.assertTrue(result['cross_width_diagnostic']['short/lane0_identical'])
            p = root/'a2.stdout'
            p.write_text('\n'.join(p.read_text(encoding='utf-8').splitlines()[:-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'record count'):
                audit.summarize(root, lock, digest)
