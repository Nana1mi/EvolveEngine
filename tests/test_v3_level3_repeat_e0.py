from copy import deepcopy
import json
from math import exp, inf, nextafter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from experiments.spark26_gguf.level3 import repeat_e0 as r
from evolveengine.v3_harness import canonical, digest


class RepeatabilityTests(TestCase):
    def rows(self):
        return [[dict(arm=arm,scenes={s:{'dev_score':100.0} for s in r.SCENES}) for arm in order]
                for order in r.ORDER]

    def test_balanced_order_and_conservative_floor(self):
        rows = self.rows()
        rows[1][1]['scenes']['W1']['dev_score'] *= exp(.02)
        result = r.summarize(rows)
        self.assertTrue(result['stable'])
        self.assertEqual(result['descriptive_log_envelope']['equal_scene'],.01)
        self.assertAlmostEqual(result['descriptive_log_envelope']['scenes']['W1'],.02)
        self.assertAlmostEqual(result['pairs'][1]['equal_scene_geomean_ratio'],exp(.005))
        self.assertEqual([first['arm'] for first,_ in rows].count('e0a'),3)

    def test_exact_summary_replay_of_pinned_gb10_receipts(self):
        # On Python 3.11 built-in sum differs by one ULP for chat pairs 3 and 4.
        # Check every summary field, including the older scoring modes.
        fixtures = {
            'independent-repeatability-result-20261125.json':
                '7f2dbdba130c6b0815e6b7679c55cf69ba0bfa59de23727c428345e761bc3833',
            'natural-eos-repeatability-result-20261125.json':
                '024707b6d52e1d6bf8033a5421734ee29aabec22f66db739398c5183f36aba75',
            'thinking-chat-repeatability-result-20261125.json':
                '76cc7d6ed28ce25f978bb650dcf2f338b6b007194c37d20a7e6eedf5dae78d38',
        }
        evidence = r.HERE / 'evidence'
        if not any((evidence / name).exists() for name in fixtures):
            self.skipTest('requires independently retained GB10 receipts; no runtime evidence ships in Git')
        for name,pin in fixtures.items():
            with self.subTest(name=name):
                raw = (evidence/name).read_bytes()
                self.assertEqual(digest(raw), pin)
                doc = json.loads(raw)
                observed = r.summarize(doc['receipts'])
                self.assertEqual(observed, {key:doc[key] for key in observed})

    def test_controller_accepts_only_bounded_libm_roundoff(self):
        import sys
        sys.path.insert(0, str(r.HERE))
        try:
            from launch_live import checked_repeatability
        finally:
            sys.path.pop(0)
        rows = self.rows()
        rows[0][1]['scenes']['W3']['dev_score'] *= exp(.02)
        exact = r.summarize(rows)
        saved = deepcopy(exact) | {'receipts': rows}
        saved['pairs'][0]['scenes_log_ratio']['W3'] = nextafter(exact['pairs'][0]['scenes_log_ratio']['W3'], inf)
        self.assertEqual(checked_repeatability(saved), exact)
        for key, value in (('W3', .021), ('W3', float('nan')), ('W3', True), ('extra', .02)):
            drift = deepcopy(saved)
            drift['pairs'][0]['scenes_log_ratio'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                checked_repeatability(drift)
        drift = deepcopy(saved)
        drift['pairs'][0]['pair'] = True  # bool compares equal to 1 in Python
        with self.assertRaises(ValueError): checked_repeatability(drift)
        with self.assertRaises(ValueError): checked_repeatability(saved | {'stable': False})
        drift = deepcopy(saved)
        drift['descriptive_log_envelope']['scenes']['W3'] += .001
        with self.assertRaises(ValueError): checked_repeatability(drift)
        drift = deepcopy(saved)
        drift['receipts'][0][1]['scenes']['W3']['dev_score'] *= 1.05
        with self.assertRaises(ValueError): checked_repeatability(drift)

    def test_marginal_scene_or_pooled_drift_blocks_envelope(self):
        for change, scenes in ((.051,('W3',)),(.031,r.SCENES)):
            rows = self.rows()
            for s in scenes:
                rows[0][1]['scenes'][s]['dev_score'] *= exp(change)
            result = r.summarize(rows)
            self.assertFalse(result['stable'])
            self.assertIsNone(result['descriptive_log_envelope'])

    def test_no_summary_for_partial_reordered_or_invalid_evidence(self):
        rows = self.rows()
        bad = deepcopy(rows)
        bad[3][1]['scenes']['W3']['dev_score']=float('nan')
        back = deepcopy(rows)
        back[2].reverse()
        for data in (rows[:-1],bad,back):
            with self.assertRaises(ValueError):
                r.summarize(data)

    def test_registration_needs_both_cold_builds_and_bound_quality(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def write(name, doc):
                raw = canonical(doc)+b'\n'
                (root/name).write_bytes(raw)
                return digest(raw)
            unchanged = dict(archive='/pinned-source',build_image='native',build_image_id='native-id',
                             limits={},native_backend_sha256='a'*64,boot_id='boot',gpu_uuid='gpu',lock='/gpu-lock')
            cfg = dict(node_ms=dict(develop_measure=300,develop_quality=200), **unchanged)
            pin = digest(canonical(cfg)+b'\n')
            write('registration.json',dict(sha256=pin,config=cfg))
            p = root/'pair.json'
            engine = 'a'*64
            ids = [str(i)*64 for i in range(1,7)]
            ph = write(p.name,dict(kind=r.control_run.PAIR_KIND,receipts=ids,engine=engine,candidate_residency_gate=False))
            saved = {}
            for i, arm in enumerate(('e0a','e0b')):
                for j, (op, status) in enumerate((('develop_build','built'),('develop_quality','valid'),('develop_measure','observed'))):
                    obj = dict(operation=op,status=status,evidence=ids[3*i+j],engine=engine,build=ids[3*i] if j else None)
                    f = f'{arm}-{op}.json'
                    write(f,obj)
                    saved[f] = obj
            self.assertEqual(r.draft(cfg,pin,p,ph)['max_total_node_ms'],3600)
            prior_path = root/'prior.json'
            self.assertEqual(write(prior_path.name,cfg),pin)
            prior = dict(path=str(prior_path),sha256=pin)
            text_cfg = dict(cfg,timing_text={'seed':'b'*64}, launch_profile='decode_graph')
            text_pin = digest(canonical(text_cfg)+b'\n')
            draft = r.draft(text_cfg,text_pin,p,ph,build_registration=prior)
            self.assertEqual(draft['build_registration'],prior)
            self.assertEqual(draft['max_total_node_ms'],4000)
            with self.assertRaisesRegex(ValueError,'same registered build inputs'):
                r.draft(dict(text_cfg,gpu_uuid='other'),text_pin,p,ph,build_registration=prior)
            with self.assertRaisesRegex(ValueError,'distinct real-text'):
                r.draft(dict(cfg,launch_profile='decode_graph'),text_pin,p,ph,build_registration=prior)

            f = 'e0b-develop_quality.json'
            saved[f]['build'] = ids[0]
            write(f,saved[f])
            with self.assertRaisesRegex(ValueError,'unrelated E0'):
                r.draft(cfg,pin,p,ph)
