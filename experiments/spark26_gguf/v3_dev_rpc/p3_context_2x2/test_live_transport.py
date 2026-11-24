"""No-network verification of the exact live execution path with a substituted wire."""
import argparse
from dataclasses import replace
from pathlib import Path
import unittest

import diagnostic as d
import live_transport as live
from test_diagnostic import ARGS, response


class Acceptance(unittest.TestCase):
    def compare(self, replies, counter=None):
        counter = counter or COUNTER
        calls, events = [], []
        fake = iter(replies)
        def record(event, row):
            events.append((event, row.copy()))
        def send(wire):
            self.assertEqual(events[-1][0], 'attempt_committed_before_dispatch')
            self.assertEqual(events[-1][1]['request_sha256'], d.digest(wire))
            calls.append(d.digest(wire))
            return next(fake)
        result = live.execute(BUNDLE, counter, send, record)
        ref = d.replay(BUNDLE, counter, d.MockUpstream(replies))
        for field in ('stop','decision','allocated_input','allocated_output','charged_input','charged_output'):
            self.assertEqual(result[field], ref[field], field)
        for actual, expected in zip(result['rows'], ref['rows']):
            self.assertEqual({k:v for k,v in actual.items() if k != 'wall_ms'}, expected)
        self.assertEqual(calls, ref['dispatched_request_sha256'])
        self.assertEqual(result['attempted_paid_calls'], len(calls))
        self.assertEqual(sum(e=='settled_once' for e,_ in events), len(calls))
        return result

    def test_three_categories_and_one_success(self):
        r = self.compare([response(), response('length',args='{"partial'),
                          response('tool_calls',args=ARGS), response()])
        self.assertEqual(r['attempted_paid_calls'],4)
        self.assertEqual(r['decision']['eligible_cell'],'max-short')

    def test_no_selection_if_every_cell_truncated(self):
        r = self.compare([response('length',m=4000,reasoning=4000)]*4)
        self.assertEqual(r['decision']['outcome'],'stop_parameter_screen_shrink_task')

    def test_halts_on_fingerprint_missing_usage_or_overrun(self):
        for bad, status in ((response(fp='different'),'fingerprint_mismatch'),
                            (response(metered=False),'usage_uncertain'),
                            (response(n=9800),'usage_uncertain'),
                            (response(m=4001),'usage_uncertain')):
            with self.subTest(status=status):
                r = self.compare([response(),bad,response(),response()])
                self.assertEqual(r['attempted_paid_calls'],2)
                self.assertEqual(r['stop'],status)

    def test_reservation_failed_does_not_send(self):
        class Count:
            tokenizer = COUNTER.tokenizer
            def reservation(self, obj):
                if obj['reasoning_effort'] == 'max':
                    return dict(rendered_tokens=15000, margin_tokens=1500,local_candidate_bound=16500)
                return COUNTER.reservation(obj)
        r = self.compare([response(),response()], Count())
        self.assertEqual(r['attempted_paid_calls'],1)
        self.assertEqual(r['rows'][1]['charged_input'],0)
        self.assertEqual(r['stop'],'reservation_failed')

    def test_unreadable_transport_is_charged_once_and_not_retried(self):
        calls = []
        def send(wire):
            calls.append(d.digest(wire))
            raise live.TransportFailure('upstream_http', 429)
        r = live.execute(BUNDLE, COUNTER, send, lambda *_: None)
        self.assertEqual(len(calls),1)
        self.assertEqual(r['rows'][0]['http_status'],429)
        self.assertEqual(r['charged_input'],2948)
        self.assertEqual(r['charged_output'],4000)
        self.assertEqual(r['stop'],'usage_uncertain')

    def test_last_request_tampering_prevents_any_attempt(self):
        bad = replace(BUNDLE, files=dict(BUNDLE.files, **{d.ORDER[-1]+'.request.json':b'x'}))
        with self.assertRaisesRegex(d.ProtocolError, 'unapproved_request_bytes'):
            live.execute(bad, COUNTER, lambda _: self.fail('dispatch'), lambda *_: self.fail('attempt'))

    def test_attempt_journal_failure_prevents_send(self):
        def bad_record(*_): raise OSError('unavailable disk')
        with self.assertRaises(OSError):
            live.execute(BUNDLE, COUNTER, lambda _: self.fail('dispatch'), bad_record)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets', required=True, type=Path)
    args = cli.parse_args()
    BUNDLE, COUNTER = d.verify(args.assets,live.FROZEN_SHA)
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance))
    raise SystemExit(not r.wasSuccessful())
