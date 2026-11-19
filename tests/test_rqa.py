"""v2.2 §8.1: wall-clock E/M starts and one paired observation per held-out lineage."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest
from unittest.mock import patch

from evolveengine.controller import Controller, H0, Lane
from evolveengine.experiment import MOCK_PROTOCOL
from evolveengine.journal import Interrupted, Journal
from evolveengine.mock import MockBackend, RULE
from evolveengine.partitions import Partitions
from evolveengine.reports import interval
from evolveengine.rqa import build_rqa_starts, seal_rqa, evaluate_rqa, summarize_rqa
from evolveengine.schema import Checkpoint, Cost, Result


KEYS = Partitions(17, b'd' * 32, final_salt=b'q' * 32)
ENV = {'simulated': True, 'backend': MockBackend.version}
NODES = ('n0', 'n1', 'n2', 'n3')
MIN = 60_000


class WallStarts(unittest.TestCase):
    def test_55_min_failure_and_90_min_boundary_reach_both_arms_not_an_inflight_step(self):
        now = [0]

        class Clock(MockBackend):
            count = 0

            def run(self, stage, request, cap, node):
                if stage == 'implement' and self.count < 2:
                    now[0] = (55, 90)[self.count] * MIN
                    self.count += 1
                    return Result({}, cap, 'failed')
                return super().run(stage, request, cap, node)

        lane = Lane('rq-a/replicate-0/mother', 'research_overhead', 'n0', Cost(120 * MIN, 1000, 10000, blocks=10))
        with TemporaryDirectory() as d, patch('evolveengine.journal.time.time_ns', side_effect=lambda: now[0] * 10**6):
            b = Clock()
            with Journal(Path(d) / 'j', enforce_wall=True) as j:
                c = Controller(j, b, KEYS.development_view(), ENV)
                points = build_rqa_starts(c, lane, b.initial_engine(), steps=2)
                self.assertEqual([p['percent'] for p in points], [50, 75])
                self.assertEqual([p['wall_ms'] // MIN for p in points], [60, 90])
                self.assertEqual([p['completed_wall_ms'] // MIN for p in points], [55, 90])
                for i, point in enumerate(points):
                    start = Checkpoint(**point['checkpoint'])
                    self.assertEqual(start.engine_digest, c.checkpoint(b.initial_engine(), lane.path).engine_digest)
                    history = j.get('memory', start.memory_digest)
                    self.assertEqual([j.get('experience', h)['outcome'] for h in history], ['implement_failed'] * (i + 1))
                    self.assertEqual(j.get_event(point['source_event'])['checkpoint'], point['checkpoint'])
                    fork = Lane('test/forks', 'eval', 'n0', MOCK_PROTOCOL.evaluation)
                    c.paired_steps(fork, str(i), start, c.baseline, c.baseline, MOCK_PROTOCOL.causal, 1)
                    for arm in ('parent', 'proposed'):
                        ready = j.get_event(f'test/forks/{i}/{arm}/step-0/ready')['checkpoint']
                        self.assertEqual(ready['memory_digest'], start.memory_digest)
                spent = j.spent(lane.path)
            now[0] += 200 * MIN
            with Journal(Path(d) / 'j', enforce_wall=True) as j:
                c = Controller(j, b, KEYS.development_view(), ENV)
                self.assertEqual(build_rqa_starts(c, lane, b.initial_engine(), 2), points)
                self.assertEqual(j.spent(lane.path), spent)

    def test_early_stop_and_no_completed_step_keep_both_points(self):
        lane = Lane('rq-a/replicate-0/mother', 'research_overhead', 'n0', MOCK_PROTOCOL.bank)
        for steps in (0, 1):
            with self.subTest(steps=steps), TemporaryDirectory() as d, Journal(Path(d) / 'journal') as j:
                backend = MockBackend()
                c = Controller(j, backend, KEYS.development_view(), ENV)
                first, second = build_rqa_starts(c, lane, backend.initial_engine(), steps)
                self.assertEqual(first['checkpoint'], second['checkpoint'])
                self.assertEqual(len(j.get('memory', first['checkpoint']['memory_digest'])), steps)
                self.assertNotEqual(first['wall_ms'], second['wall_ms'])

    def test_crash_after_admission_does_not_backdate_recoverable_memory(self):
        now = [0]

        class Clock(MockBackend):
            def _measure(self, req):
                now[0] = 48 * MIN
                return super()._measure(req)
        lane = Lane('rq-a/replicate-0/mother', 'research_overhead', 'n0', Cost(120 * MIN, 1000, 10000, blocks=10))
        with TemporaryDirectory() as d, patch('evolveengine.journal.time.time_ns', side_effect=lambda: now[0] * 10**6):
            b = Clock()
            with Journal(Path(d) / 'j', enforce_wall=True) as j:
                c = Controller(j, b, KEYS.development_view(), ENV)
                write = j.event

                def stop(key, value):
                    if key == lane.path + '/step-0/finished':
                        raise Interrupted(key)
                    write(key, value)
                with patch.object(j, 'event', side_effect=stop), self.assertRaises(Interrupted):
                    build_rqa_starts(c, lane, b.initial_engine(), 1)
            now[0] = 70 * MIN
            with Journal(Path(d) / 'j', enforce_wall=True) as j:
                c = Controller(j, b, KEYS.development_view(), ENV)
                a, bp = build_rqa_starts(c, lane, b.initial_engine(), 1)
                self.assertEqual(j.get('memory', a['checkpoint']['memory_digest']), [])
                self.assertEqual(bp['completed_wall_ms'], 70 * MIN)
                self.assertNotEqual(a['checkpoint']['engine_digest'], bp['checkpoint']['engine_digest'])
                self.assertEqual(j.spent(lane.path).blocks, 2)


class FrozenEndpoint(unittest.TestCase):
    def make_report(self, d, n=0, policy=H0):
        b = MockBackend()
        with Journal(Path(d) / f'search{n}') as j:
            bundle = seal_rqa(j, b, KEYS.development_view(), MOCK_PROTOCOL, NODES, n, ENV, policy)
            self.assertFalse(any(row['id'].endswith('/propose') for row in j.budget_rows()))
            for p in bundle['payload']['points']:
                origin = p['start']['checkpoint']
                for end in p['submissions'].values():
                    hist = j.get('memory', end['checkpoint']['memory_digest'])
                    prefix = j.get('memory', origin['memory_digest'])
                    self.assertEqual(hist[:len(prefix)], prefix)
                self.assertEqual(origin['lineage'], bundle['payload']['run_manifest']['mother_lineage'])
        with Journal(Path(d) / f'judge{n}') as j:
            result = evaluate_rqa(j, b, KEYS.final_view(), MOCK_PROTOCOL, bundle, bundle['digest'])
            prior = j.spent('rq-a')
            self.assertEqual(evaluate_rqa(j, b, KEYS.final_view(), MOCK_PROTOCOL, bundle, bundle['digest']), result)
            self.assertEqual(j.spent('rq-a'), prior)
        return result

    def test_equal_engines_are_zero_and_cannot_masquerade_as_a_real_effect(self):
        with TemporaryDirectory() as d:
            reports = [self.make_report(d, n) for n in (0, 1)]
            result = summarize_rqa(reports)
            self.assertEqual(result['paired_log_differences'], [0., 0.])
            self.assertEqual(result['ci95'], [0., 0.])
            self.assertFalse(result['positive_effect'])
            self.assertFalse(result['confirmatory_pass'])
            with self.assertRaises(ValueError):
                summarize_rqa([reports[0], reports[0]])
            changed = self.make_report(d, 2, {'rules': [RULE]})
            with self.assertRaises(ValueError):
                summarize_rqa([reports[0], changed])

    def test_sub_three_percent_positive_uses_both_points_not_block_or_stage_count(self):
        with TemporaryDirectory() as d:
            reports = [self.make_report(d, n, {'rules': [RULE]}) for n in (0, 1)]
            for r in reports:
                # A test fixture: different Q/T throughput levels, and misleading v1 speedups.
                for pt, ratio in zip(r['points'], (1., 1.04)):
                    for label, rate in [('parent', 100.), ('proposed', 100. * ratio)]:
                        pt['blind']['engines'][label] = {'status': 'valid', 'speedup': 99.,
                            'tokens_per_second': [{'candidate': rate, 'reference': 70.}] * MOCK_PROTOCOL.final_blocks}
            expected = math.log(1.04) / 2
            out = summarize_rqa(reports)
            self.assertEqual(out['n'], 2)
            self.assertAlmostEqual(out['paired_log_differences'][0], expected)
            self.assertTrue(out['positive_effect'])
            self.assertFalse(out['at_least_3pct_point_estimate'])
            self.assertFalse(out['confirmatory_pass'])  # The inputs are marked simulated.
            self.assertFalse(interval([expected] * 2)['meets_design_threshold'])  # v1 is unchanged.
            missing = deepcopy(reports[0])
            missing['points'].pop()
            with self.assertRaises(ValueError):
                summarize_rqa([missing])
            bad = deepcopy(reports[0])
            bad['points'][0]['blind'] = {'status': 'evaluation_budget_exhausted'}
            out = summarize_rqa([bad, reports[1]])
            self.assertEqual(out['n'], 1)
            self.assertEqual(len(out['incomplete_lineages']), 1)
            self.assertFalse(out['confirmatory_pass'])

    def test_fallback_uses_measured_start_and_invalid_reference_is_not_a_measured_zero(self):
        with TemporaryDirectory() as d:
            r = self.make_report(d)
            for pt in r['points']:
                rows = pt['blind']['engines']
                rows['parent'] = {'status': 'fallback_quality', 'speedup': 1.}
                rows['proposed'] = {'status': 'valid', 'tokens_per_second':
                                    [{'candidate': 105.}] * MOCK_PROTOCOL.final_blocks}
                rows['start'] = {'status': 'valid', 'tokens_per_second':
                                 [{'candidate': 100.}] * MOCK_PROTOCOL.final_blocks}
            self.assertAlmostEqual(summarize_rqa([r])['paired_log_differences'][0], math.log(1.05))
            r['points'][1]['blind'] = {'status': 'invalid_reference'}
            out = summarize_rqa([r])
            self.assertEqual(out['n'], 0)
            self.assertEqual(len(out['incomplete_lineages']), 1)
            self.assertFalse(out['positive_effect'])
