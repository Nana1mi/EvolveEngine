from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from experiments.spark26_gguf.level3 import calibrate_isolation_cache as cold
from evolveengine.v3_level3_collect import source_bytes


class ColdObserverCalibrationTests(TestCase):
    def test_only_cancellation_capture_is_instrumented(self):
        src=source_bytes(Path(cold.__file__).with_name('observe_trajectory.py'))
        got=cold.instrument(src)
        compile(got,'cold_trajectory','exec')
        self.assertEqual(got.split(b'def teach(')[1],src.split(b'def teach(')[1])
        self.assertIn(b'call=survivor_scope.call',got)
        self.assertIn(b'call=neighbor_scope.call',got)
        self.assertIn(b'cache_parity=cold.witness(q.TRIGGERS)',got)
        self.assertNotIn(b'ColdPairs',src)
        with self.assertRaisesRegex(ValueError,'anchor changed'):
            cold.instrument(got)

    def test_missing_misordered_or_warm_witness_cannot_calibrate(self):
        with patch.object(cold.q,'TRIGGERS',(8,)):
            rows=[dict(trigger=8,arm=a,cached_tokens_initial=[0,0]) for a in ('cancel','complete')]
            self.assertEqual(cold.verify_cache(dict(cache_parity=rows)),rows)
            for bad in (None,rows[::-1],rows[:1],rows+[rows[0]],
                        [dict(rows[0],cached_tokens_initial=[0,1]),rows[1]],
                        [dict(rows[0],cached_tokens_initial=[0,False]),rows[1]]):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    cold.verify_cache(dict(cache_parity=bad))

    def test_shadow_lane_cannot_sign_live_quality(self):
        lane=object.__new__(cold.ColdLane)
        for operation in (lane.activate,lane.screened):
            with self.assertRaisesRegex(ValueError,'cannot sign'):
                operation()


class ColdObserverSummaryTests(TestCase):
    def rows(self):
        from experiments.spark26_gguf.level3 import summarize_isolation_cache as sums
        return [dict(configuration=mode,candidate=trial,teacher=teacher,node_ms=100,
          metrics=dict(reference=dict(protocol=cold.q.VERSION,maxima={k:0. for k in cold.q.METRICS}),
          cache=[dict(trigger=k,arm=a,cached_tokens_initial=[0,0]) for k in cold.q.TRIGGERS for a in ('cancel','complete')],
          points=[dict({metric:(None if metric=='paired_draw_probability' else .01) for metric in sums.METRICS},
                            trigger=k,step=i,same_prefix=False,same_draw=False) for k in cold.q.TRIGGERS for i in cold.q.TAIL]))
          for mode,trial,teacher in cold.ORDER]

    def test_missing_draw_counts_are_not_zero_variance(self):
        from experiments.spark26_gguf.level3.summarize_isolation_cache import summary
        rows=self.rows(); info=summary(rows)
        self.assertEqual(set(info),{'registered','radix_bs6'})
        self.assertIsNone(info['registered']['maxima']['paired_draw_probability'])
        self.assertEqual(info['registered']['coverage_by_metric']['paired_draw_probability'],0)
        self.assertEqual(info['registered']['coverage_by_metric']['paired_eos_logp'],8)
        probe=rows[0]['metrics']['points'][0]
        probe.update(same_draw=True,paired_draw_probability=.014)
        self.assertEqual(summary(rows)['registered']['maxima']['paired_draw_probability'],.014)

    def test_summary_rejects_misbound_receipts_and_invented_zero(self):
        from experiments.spark26_gguf.level3.summarize_isolation_cache import summary
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'order'): summary(rows[::-1])
        with self.assertRaisesRegex(ValueError,'incomplete'): summary(rows[:-1])
        rows[0]['metrics']['points'][0]['paired_draw_probability']=0.
        with self.assertRaisesRegex(ValueError,'unbound'): summary(rows)
