from copy import deepcopy
from itertools import count
from math import exp, log
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from random import Random

from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_level3_workload import profile_inputs, input_fingerprints


class TrajectoryTests(TestCase):
    def setUp(self):
        self.enterContext(patch.object(q,'VOCAB',64)); self.enterContext(patch.object(q,'MARKERS',(1,4,42)))
        ids=count(); self.rid=lambda:f'{next(ids):032x}'
        base=-log(64); self.base=base
        self.vector=q.pack([[base,t] for t in range(64)])
        self.c=dict(protocol=q.VERSION,role='candidate',decode=[],isolation=[])
        for case in q.CASES:
            self.c['decode'].append(dict(rid=self.rid(),case=case,prompt_sha=q.ih(q.prefix(case)),input=len(q.prefix(case)),
                ids=[7]*q.STEPS,cap=q.STEPS,finish='length',vectors=[self.vector]*q.STEPS))
        for trigger in q.TRIGGERS:
            pair=dict(trigger=trigger)
            for arm in ('complete','cancel'):
                def stream(s):
                    n=q.SURVIVOR if s else q.NEIGHBOR if arm=='complete' else trigger+1
                    v=7 if arm=='complete' else 8  # legal tie, diverging survivor not a batch-invariance failure
                    return dict(rid=self.rid(),cap=q.SURVIVOR if s else q.NEIGHBOR,prompt_sha=q.ih(q.prefix(7 if s else 6)),
                        ids=[v]*n,probs=[[base]*4]*n if s else [], ack=None if s or arm=='complete' else [trigger,trigger*10+1,trigger*10+3,200],
                        progress=[[i,i*10] for i in range(1,n+1)],done=True,finish='length' if s or arm=='complete' else 'abort')
                pair[arm]=[stream(True),stream(False)]
            self.c['isolation'].append(pair)
        self.teacher=dict(role='teacher',protocol=q.VERSION,rows=[])
        for j in q.teacher_plan(self.c):
            row=dict(key=j['key'],rid=self.rid(),prompt_sha=q.ih(j['ids']),input=len(j['ids']),cap=1,finish='length',generated=7)
            if j['wanted']=='all': row['vector']=self.vector
            else: row['values']=[base]*len(j['wanted'])
            self.teacher['rows'].append(row)
        r=q.compare(self.c,self.teacher); self.limits=q.envelope([r,r])

    def test_e0_uses_actual_candidate_prefix_and_never_rolls_out(self):
        plan=q.teacher_plan(self.c)
        self.assertEqual(plan[3]['ids'],q.prefix(q.CASES[0])+[7]*3)
        self.assertNotIn('vectors',plan[3]); self.assertEqual(plan[3]['wanted'],'all')
        report=q.compare(self.c,self.teacher)
        self.assertEqual(q.verdict(report,self.limits)['status'],'valid')
        self.assertEqual(sum(d['path']=='autoregressive_step' for d in report['decode_positions']),len(q.CASES)*(q.STEPS-1))
        self.c['decode'][0]['ids'][0]=8
        with self.assertRaisesRegex(ValueError,'prefix'): q.compare(self.c,self.teacher)

    def test_probabilities_wrong_only_in_decode_or_after_abort_are_detected(self):
        self.c['decode'][0]['vectors']=self.c['decode'][0]['vectors'][:]
        # A modest transfer to EOS, not an early stop or changed generated token count.
        scale=.001; p=1/64
        self.c['decode'][0]['vectors'][2]=q.pack([(log(p+scale if t==1 else p-scale if t==2 else p),t) for t in range(64)])
        self.assertIn('decode_eos_logp',q.verdict(q.compare(self.c,self.teacher),self.limits)['failed'])
        self.c['isolation'][0]['cancel'][0]['probs'][-1]=[self.base+1,self.base,self.base,self.base]
        failed=q.verdict(q.compare(self.c,self.teacher),self.limits)['failed']
        self.assertIn('iso_paired_residual',failed); self.assertIn('iso_eos_logp',failed)

    def test_isolation_diagnostics_distinguish_common_drift_interaction_and_draw(self):
        from experiments.spark26_gguf.level3.analyze_isolation_residual import decompose
        i=q.TAIL[0]
        pair=self.c['isolation'][0]
        for arm in ('complete','cancel'):
            pair[arm][0]['ids'][:i]=[7]*i
            pair[arm][0]['probs'][i][0]=self.base+1
        pair['complete'][0]['probs'][i][-1]=self.base+2
        for row, job in zip(self.teacher['rows'],q.teacher_plan(self.c),strict=True):
            row['prompt_sha']=q.ih(job['ids'])
        before=deepcopy((self.c,self.teacher))
        plain=decompose(self.c,self.teacher,self.limits)
        split=decompose(self.c,self.teacher,self.limits,separate_draw=True)
        row=split['points'][0]
        self.assertEqual(plain['verdict'],split['verdict'])
        self.assertTrue(row['same_survivor_prefix'])
        self.assertFalse(row['same_drawn_token'])
        self.assertIsNone(row['paired_drawn_probability_residual_abs'])
        self.assertEqual(row['common_eos_log_residual'],1.)
        self.assertEqual(row['paired_eos_log_residual_abs'],0.)
        self.assertEqual(row['paired_marker_residual_max'],0.)
        self.assertTrue(row['complete']['drawn_probability_strictly_dominates'])
        self.assertFalse(row['cancel']['drawn_probability_strictly_dominates'])
        pair['cancel'][0]['probs'][i][0]=self.base+1.25
        row=decompose(self.c,self.teacher,self.limits,separate_draw=True)['points'][0]
        self.assertAlmostEqual(row['paired_eos_log_residual_abs'],.25)
        self.assertGreater(row['paired_marker_residual_max'],0.)
        pair['cancel'][0]['probs'][i][0]=self.base+1
        self.assertEqual((self.c,self.teacher),before)
        pair['cancel'][0]['ids'][i]=pair['complete'][0]['ids'][i]
        for row, job in zip(self.teacher['rows'],q.teacher_plan(self.c),strict=True):
            row['prompt_sha']=q.ih(job['ids'])
        row=decompose(self.c,self.teacher,self.limits,separate_draw=True)['points'][0]
        self.assertTrue(row['same_drawn_token'])
        self.assertAlmostEqual(row['paired_drawn_probability_residual_abs'],(exp(2)-1)/64)

    def test_cold_matrix_separates_absolute_quality_and_cancellation_evidence(self):
        from experiments.spark26_gguf.level3.calibrate_isolation_cache import compare
        self.c['cache_parity']=[dict(trigger=k,arm=a,cached_tokens_initial=[0,0])
                                for k in q.TRIGGERS for a in ('cancel','complete')]
        tail=q.TAIL[0]; pair=self.c['isolation'][0]
        # The arms deliberately use different draws (a legitimate tie).
        self.assertNotEqual(pair['complete'][0]['ids'][tail],pair['cancel'][0]['ids'][tail])
        # Changes shared by both arms must not count as a cancellation effect.
        for a in ('complete','cancel'):
            pair[a][0]['probs'][tail][0]=self.base+1.
        result=compare(self.c,self.teacher); point=result['points'][0]
        self.assertEqual(point['abs_eos_logp'],1.)
        self.assertEqual(point['paired_eos_logp'],0.)
        self.assertIsNone(point['paired_draw_probability'])
        self.assertFalse(point['same_draw'])
        # The new observation records the distinct causal contrast without
        # lowering the existing absolute-quality requirement.
        pair['cancel'][0]['probs'][tail][0]=self.base+0.5
        point=compare(self.c,self.teacher)['points'][0]
        self.assertEqual(point['abs_eos_logp'],1.)
        self.assertEqual(point['paired_eos_logp'],0.5)
        self.assertGreater(point['paired_marker_probability'],0.)

    def test_late_distribution_probe_never_passes_missing_mass_as_zero(self):
        from experiments.spark26_gguf.level3 import late_trajectory as late
        self.c['cache_parity']=[dict(trigger=k,arm=a,cached_tokens_initial=[0,0])
                                for k in q.TRIGGERS for a in ('cancel','complete')]
        for pair in self.c['isolation']:
            for arm in ('complete','cancel'):
                pair[arm][0]['tail_top']={str(i):[[self.base,t] for t in range(8)] for i in q.TAIL}
        for row,job in zip(self.teacher['rows'],q.teacher_plan(self.c),strict=True):
            if job['wanted']=='all': continue
            row['vector']=self.vector
            vector=q.unpack(self.vector)
            row['values']=[vector[t] for t in job['wanted']]
        with patch.object(late,'TOP_K',8):
            report=late.tail_report(self.c,self.teacher)
        self.assertEqual(report['absolute_tail_tv'],'inconclusive')
        self.assertFalse(report['live_gate'])
        self.assertGreater(report['tail_points'][0]['paired']['upper'],.8)
        self.assertLess(report['tail_points'][0]['paired']['lower'],1e-7)
        self.assertEqual(len(report['tail_points']),len(q.TRIGGERS)*len(q.TAIL))
        # Two distinct continuations are still covered when we observe enough mass.
        for pair in self.c['isolation']:
            for arm in ('complete','cancel'):
                pair[arm][0]['tail_top']={str(i):[[self.base,t] for t in range(62)] for i in q.TAIL}
        with patch.object(late,'TOP_K',62):
            report=late.tail_report(self.c,self.teacher)
        self.assertEqual(report['absolute_tail_tv'],'certified')
        self.assertLess(report['tail_points'][0]['paired']['upper'],.05)

    def test_fault_replay_uses_copies_and_attempts_both_independent_groups(self):
        from experiments.spark26_gguf.level3 import late_trajectory as late, replay_late_trajectory as replay
        self.c['cache_parity']=[dict(trigger=k,arm=a,cached_tokens_initial=[0,0])
                                for k in q.TRIGGERS for a in ('cancel','complete')]
        for pair in self.c['isolation']:
            for a in ('complete','cancel'):
                pair[a][0]['tail_top']={str(i):[[self.base,t] for t in range(q.VOCAB)] for i in q.TAIL}
        vector=q.unpack(self.vector)
        for row,job in zip(self.teacher['rows'],q.teacher_plan(self.c),strict=True):
            if job['wanted']=='all': continue
            row['vector']=self.vector; row['values']=[vector[t] for t in job['wanted']]
        baseline=deepcopy((self.c,self.teacher))
        with patch.object(late,'TOP_K',q.VOCAB):
            report=late.tail_report(self.c,self.teacher)
            results=replay.check_controls(self.c,self.teacher,report,self.limits)
        self.assertEqual(results['baseline']['status'],'valid')
        self.assertEqual(results['synthetic_controls']['common_eos']['groups']['mixed_tail_absolute']['status'],'violation')
        self.assertEqual(results['synthetic_controls']['paired_eos']['groups']['paired_cancel_numeric']['status'],'violation')
        self.assertEqual((self.c,self.teacher),baseline)

    def test_incomplete_candidate_only_does_not_start_teacher_or_quarantine(self):
        from evolveengine.v3_level3_trajectory_lane import candidate_jobs
        from evolveengine.v3_level3_worker import ObservationUnavailable
        self.assertEqual(candidate_jobs(self.c), q.teacher_plan(self.c))
        for bad in ({}, {'sha256':'1'*64,'f32le_zlib':'$'}, {'sha256':'0'*64,'f32le_zlib':'YWJj'}):
            self.c['decode'][0]['vectors'][0]=bad
            with self.assertRaises(ObservationUnavailable): candidate_jobs(self.c)

    def test_fault_replay_needs_the_pinned_clean_observations_and_never_mutates_them(self):
        from experiments.spark26_gguf.level3.replay_trajectory_controls import controls
        baseline=deepcopy(self.c);metrics=q.compare(self.c,self.teacher)
        result=controls(self.c,self.teacher,metrics,self.limits)
        self.assertIn('decode_eos_logp',result['decode_only']);self.assertIn('iso_paired_residual',result['post_cancel_only'])
        self.assertEqual(self.c,baseline)
        with self.assertRaisesRegex(ValueError,'signed|reproduce|E0 maxima|E0'):
            controls(self.c,self.teacher,{},self.limits)

    def test_no_fake_cancel_reference_or_partial_vocab(self):
        for change in ('ack','neighbor','overlap','rid','vector','vocab'):
            candidate=deepcopy(self.c); a=candidate['isolation'][0]
            if change=='ack': a['cancel'][1]['ack'][-1]=503
            elif change=='neighbor': a['complete'][1]['prompt_sha']='0'*64
            elif change=='overlap': a['cancel'][1]['progress']=[[n,99999+t] for n,t in a['cancel'][1]['progress']]
            elif change=='rid': a['complete'][1]['rid']=a['cancel'][1]['rid']
            elif change=='vector': candidate['decode'][0]['vectors'][0]=dict(self.vector,sha256='0'*64)
            else: candidate['decode'][0]['vectors']=[]
            with self.subTest(change=change), self.assertRaises(ValueError): q.compare(candidate,self.teacher)

    def test_dedicated_profile_windows_cannot_borrow_frozen_chat_inputs(self):
        rng=Random(512); docs=[[rng.randrange(1000) for _ in range(1500)]]; scenes={'W2':((64,16),)}
        orig=profile_inputs(docs,scenes,2,'a'*64)
        new=profile_inputs(docs,scenes,2,'a'*64,SimpleNamespace(waves=orig))
        def pins(x): return {p for ws in input_fingerprints(x).values() for w in ws for p in w}
        self.assertFalse(pins(orig)&pins(new)); self.assertEqual(len(pins(new)),2)
