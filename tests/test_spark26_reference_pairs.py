import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    from reference_pair_audit import audit, BUNDLE_NAMES, screen, LOCK_PATH


def fixture(ratios, slow=None, unequal=False):
    lock = json.loads(LOCK_PATH.read_bytes())
    from_reference = dict(n_pairs=4, timed_requests=240, pairs=[], per_run_medians={},
                          distinct_timed_token_hashes={f'{name}/lane{lane}':1 for name,shape in lock['shapes'].items()
                                                       for lane in range(shape['width'])})
    if unequal:
        from_reference['distinct_timed_token_hashes']['short1/lane0'] = 2
    for i, ratio in enumerate(ratios, 1):
        a,b = f'a{i}', f'b{i}'
        pct = 200*(ratio-1)/(ratio+1)
        from_reference['pairs'].append(dict(run_order=lock['pairs'][i-1]['labels'], signed_b_minus_a_pct={
            s: dict(wave_output_tps=pct, wave_max_ttft_ms=3 if slow == s else 0, wave_wall_ms=-pct)
               for s in lock['shapes']}))
        for name,value in ((a,100),(b,100*ratio)):
            from_reference['per_run_medians'][name] = {s:dict(wave_output_tps=value) for s in lock['shapes']}
    return lock,from_reference


class ReferencePairsTest(unittest.TestCase):
    def test_screens_gain_not_request_count(self):
        lock, data = fixture([1.03,1.04,1.03,0.999])
        self.assertTrue(screen(lock,data)['engineering_screen_passed'])
        data['n_pairs'] -= 1
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            screen(lock,data)

    def test_guard_for_token_match_and_ttft(self):
        lock, data = fixture([1.06,1.04,1.05,1.03], slow='long2',unequal=True)
        verdict = screen(lock,data)
        self.assertTrue(verdict['checks']['primary_median_exceeds_min'])
        self.assertFalse(verdict['checks']['all_timed_tokens_unchanged'])
        self.assertFalse(verdict['checks']['long2_ttft_guard'])
        self.assertFalse(verdict['engineering_screen_passed'])

    def test_weak_gain_does_not_pass(self):
        verdict = screen(*fixture([1.001,1.012,1.003,1.002]))
        self.assertTrue(verdict['checks']['positive_pairs'])
        self.assertFalse(verdict['engineering_screen_passed'])

    def test_complete_saved_evidence_and_tampered_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = LOCK_PATH.read_bytes()
            lock = json.loads(work)
            (root/LOCK_PATH.name).write_bytes(work)
            for name in BUNDLE_NAMES-{LOCK_PATH.name}:
                (root/name).write_text('script fixture\n', encoding='utf-8')

            def manifest(filename, names):
                path = root/filename
                with path.open('wb') as f:
                    for name in sorted(names):
                        f.write(f'{hashlib.sha256((root/name).read_bytes()).hexdigest()}  {name}\n'.encode())
                return hashlib.sha256(path.read_bytes()).hexdigest()

            bundle_hash = manifest('bundle.sha256', BUNDLE_NAMES)
            digest = hashlib.sha256(work).hexdigest()
            raw, events = set(), []
            ts = '2026-11-23T14:10:00+08:00'
            for pair in lock['pairs']:
                for label in pair['labels']:
                    waves = [dict(kind='run_identity', label=label, workload_sha256=digest)]
                    logs = [f'EE_REFERENCE_PAIR {label} {lock["source_sha256"][label[0]]}']
                    for shape in pair['shape_order']:
                        w = lock['shapes'][shape]['width']
                        n = lock['shapes'][shape]['input_tokens']
                        for phase, count in (('warmup',2),('timed',5)):
                            for i in range(count):
                                logs += [f'EE_CALIBRATION {shape} {phase} wave-{i}',
                                         f'Prefill batch, #new-seq: {w}, #new-token: {w*n}, #cached-token: 0,']
                                waves.append(dict(kind='wave', shape=shape, index=i, phase=phase,
                                                  wall_seconds=1 if label[0]=='a' else .95,
                                                  requests=[dict(rid=f'{label}-{shape}-{phase}-{i}-{j}',
                                                  input_tokens=n, output_tokens=48, finish='length',
                                                  first_token_ms=100, output_sha256='b'*64) for j in range(w)]))
                    (root/f'{label}.stdout').write_text('\n'.join(map(json.dumps,waves)), encoding='utf-8')
                    (root/f'{label}.server.log').write_text('\n'.join(logs+['EE_CALIBRATION complete']),encoding='utf-8')
                    for which in ('pre','post'):
                        (root/f'{label}.{which}-gpu.csv').write_text('NVIDIA GB10, 580.178.04\n', encoding='utf-8')
                    raw.update(f'{label}.{ext}' for ext in ('stdout','server.log','pre-gpu.csv','post-gpu.csv'))
                    events.extend(f'{stage}\t{label}\t{ts}' for stage in ('start','done'))
            manifest('raw-evidence.sha256',raw)
            (root/'run-order.tsv').write_text('\n'.join(events)+'\n', encoding='utf-8')
            (root/'run.closed').write_text(ts+'\t0\n', encoding='utf-8')
            (root/'post-compute.csv').write_text('', encoding='utf-8')
            self.assertTrue(audit(root,bundle_hash)['outcome']['engineering_screen_passed'])
            (root/'b4.stdout').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                audit(root,bundle_hash)
