from pathlib import Path
from unittest import TestCase

from experiments.spark26_gguf.level3 import live_split_quality as split
from evolveengine.v3_level3_trajectory_lane import SPLIT_CASES, SPLIT_PROTOCOL, trajectory_cases


class LiveSplitRegistrationTests(TestCase):
    def test_overlay_rejects_both_unregistered_drift_and_missing_changes(self):
        old={k:'old-bytes' for k in split.OVERLAY_CHANGES}|{'some-unrelated-driver':'same'}
        new={k:'live-bytes' for k in split.OVERLAY_CHANGES}|{'some-unrelated-driver':'same'}
        row={'implementation_overlay':{k:[old[k],new[k]] for k in split.OVERLAY_CHANGES}}
        self.assertIsNone(split.verify_overlay(old,new,row))
        with self.assertRaisesRegex(ValueError,'unregistered'):
            split.verify_overlay(old,new|{'some-unrelated-driver':'drift'},row)
        one=next(iter(split.OVERLAY_CHANGES))
        with self.assertRaisesRegex(ValueError,'unregistered'):
            split.verify_overlay(old,new|{one:old[one]},row)
        with self.assertRaisesRegex(ValueError,'unregistered'):
            split.verify_overlay(old,new,{'implementation_overlay':row['implementation_overlay']|{'some-unrelated-driver':['same','same']}})

    def test_shadow_status_never_counts_as_live(self):
        groups={k:dict(status='valid',violations=[],insufficient=[]) for k in SPLIT_CASES}
        row=dict(protocol=SPLIT_PROTOCOL,live_gate=False,status='valid',groups=groups)
        with self.assertRaisesRegex(ValueError,'live split'):
            trajectory_cases(row,SPLIT_PROTOCOL)
        row['live_gate']=True
        self.assertEqual(trajectory_cases(row,SPLIT_PROTOCOL),dict.fromkeys(SPLIT_CASES,'pass'))
        groups['mixed_tail_absolute']['insufficient']=['unknown']
        with self.assertRaisesRegex(ValueError,'group status'):
            trajectory_cases(row,SPLIT_PROTOCOL)
        groups['mixed_tail_absolute']['status']='insufficient_evidence'
        with self.assertRaisesRegex(ValueError,'overall status'):
            trajectory_cases(row,SPLIT_PROTOCOL)

    def test_staged_observer_code_matches_the_hardware_positive(self):
        # Public observer-code fingerprint only; no hardware receipt is a unit-test fixture.
        expected='0ff0f3e560a058644eee89158273cc56077ba852266e6aee6cb2076e97f1044e'
        from types import SimpleNamespace
        lane=split.LiveTailLane(SimpleNamespace(host=None,service_spec=None),split.scripts(split.HERE),
                                Path(split.HERE/'isolation_cache_parity.py'))
        self.assertEqual(lane.code,expected)
