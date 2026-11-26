from unittest import TestCase

from experiments.spark26_gguf.level3 import split_quality as split


class SplitQualityTests(TestCase):
    def setUp(self):
        q=split.q
        self.limits={k:.005 for k in q.METRICS}; self.limits.update(decode_tv=.05,decode_eos_logp=.40)
        coords=[(tr,i) for tr in q.TRIGGERS for i in q.TAIL]
        self.report=dict(live_gate=False,absolute_tail_tv_limit=.05,absolute_tail_tv='certified',
             legacy_observation=dict(reference=dict(protocol=q.VERSION,maxima={k:0. for k in q.METRICS}),
                 points=[dict(trigger=tr,step=i,paired_marker_probability=0.) for tr,i in coords]),
             tail_points=[dict(trigger=tr,step=i,
                 arms={a:dict(tv=dict(lower=0.,upper=.001),status='certified',eos_log_delta=0.)
                 for a in ('complete','cancel')},paired=dict(lower=0.,upper=.001)) for tr,i in coords])

    def statuses(self):
        x=split.assess(self.report,self.limits)
        self.assertFalse(x['live_gate'])
        return [x['groups'][k]['status'] for k in split.GROUPS]

    def test_common_eos_drift_is_absolute_only(self):
        self.assertEqual(self.statuses(),['valid']*3)
        pt=self.report['tail_points'][0]
        for arm in pt['arms'].values(): arm['eos_log_delta']=.41
        self.assertEqual(self.statuses(),['valid','violation','valid'])

    def test_cancellation_can_fail_even_when_both_arms_and_early_decode_pass(self):
        pt=self.report['tail_points'][0]
        pt['arms']['complete']['eos_log_delta']=-.3
        pt['arms']['cancel']['eos_log_delta']=.3
        self.assertEqual(self.statuses(),['valid','valid','violation'])

    def test_unknown_tail_and_early_numeric_violation_stay_distinct(self):
        self.report['tail_points'][0]['paired']['upper']=.08
        self.assertEqual(self.statuses(),['valid','valid','insufficient_evidence'])
        invalid=self.report['legacy_observation']['reference']['maxima']
        invalid['decode_greedy_regret']=.1
        self.assertEqual(self.statuses(),['violation','valid','insufficient_evidence'])
        self.assertEqual(split.assess(self.report,self.limits)['status'],'violation')
        self.report['tail_points'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'): split.assess(self.report,self.limits)
