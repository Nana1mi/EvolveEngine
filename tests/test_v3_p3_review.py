from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

DEV=Path(__file__).resolve().parents[1]/'experiments/spark26_gguf/v3_dev_rpc'
sys.path.insert(0,str(DEV))
from p3_protocol import exploratory_index, propose_w
from evolveengine.v3_harness import digest
from evolveengine.v3_session import DevReceipt


def scenes(n):return {k:{'score':n} for k in ('W1','W2','W3','W4')}


class P3Review(unittest.TestCase):
    def test_scores_use_current_e0(self):
        self.assertEqual(exploratory_index(scenes(120),scenes(100))['index'],1.2)
        self.assertEqual(exploratory_index(scenes(120),scenes(120))['index'],1)
        with self.assertRaises(ValueError):exploratory_index(scenes(120),scenes(0))

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional Spark audit dependency')
    def test_corrupt_live_replay_blocks_s2_instead_of_empty_facts(self):
        import p3
        arms=[];calls=[]
        def model(arm):arms.append(arm);return lambda *_:None
        def worker(op,edits,eng,base,lease,job):
            calls.append((op,edits));return DevReceipt(eng,op,'observed',100,digest(b'e0'),{'scenes':scenes(1)})
        def replay(job):
            if len(arms):raise ValueError('corrupt S1 raw archive fixture')
        with tempfile.TemporaryDirectory() as t,patch.object(p3,'sandbox',return_value={'status':'pi_session_complete','isolated_state_removed':True}):
            output=Path(t)/'p3'
            with self.assertRaisesRegex(ValueError,'corrupt S1 raw archive fixture'):
                p3.run_pair(b'def f(): return 1\n',output,model,worker,replay)
            self.assertEqual(arms,['S1']);self.assertEqual(calls,[('develop_measure',{})])
            self.assertFalse((output/'S2-E0.json').exists())
            failed=json.loads((output/'S1-to-S2.json').read_bytes())
            self.assertEqual(failed['status'],'blocked_S2_not_started');self.assertNotIn('facts',failed)
            self.assertEqual(json.loads((output/'W-budget-proposal.json').read_bytes())['status'],'hold_requires_settled_uncensored_costs')

    def test_cost_rule_includes_runs_without_deliveries_or_harness_changes(self):
        row=dict(status='completed',wall_ms=200000,node_ms=250000,usage={'input':11000,'output_including_reasoning':2500},workers=[],explicitly_delivered=None,budget_stop=False)
        p=propose_w([row,row]);self.assertEqual(p['status'],'cost_proposal_not_frozen');self.assertEqual(p['training_total']['gpu_node_seconds'],3180)
        self.assertEqual(p['training_total']['development_calls'],13)
        self.assertEqual(propose_w([row,dict(row,budget_stop=True)])['status'],'hold_requires_settled_uncensored_costs')
