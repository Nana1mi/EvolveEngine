"""Mock-SSE acceptance; no network, subprocess, Agent, GPU or credential access."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import diagnostic as d


def packet(delta=None, end=None, meter=None, fp=d.FINGERPRINT, model=d.MODEL, standalone=False):
    return b'data: ' + d.canonical(dict(id='offline', object='chat.completion.chunk', created=1,
        model=model, system_fingerprint=fp,
        choices=[] if standalone else [dict(index=0, delta=delta or {}, finish_reason=end)],
        usage=meter)) + b'\n\n'


ARGS = dict(path=d.TARGET, function=d.FUNCTION, expression='min(100, max(value, 0))')
DONE = b'data: [DONE]\n\n'


def response(finish='stop', *, args=None, wrong_tool=False, second_call=False, fp=d.FINGERPRINT,
             model=d.MODEL, n=100, m=50, reasoning=40, metered=True, usage_only_chunk=False,
             end_marker=True, ambiguous=False):
    def event(**kwargs):
        return packet(**kwargs, fp=fp, model=model)
    stream = event(delta=dict(role='assistant', reasoning_content='synthetic reasoning '))
    stream += event(delta=dict(reasoning_content='prefix'))
    if args is not None:
        encoded = args if isinstance(args,str) else d.canonical(args).decode()
        _id = 'offline_call'
        delta = dict(tool_calls=[dict(index=0, id=_id, type='function', function=dict(
                     name='read' if wrong_tool else d.PROPOSE, arguments=encoded[:12]))])
        if ambiguous:
            delta['reasoning_content'] = 'same event ambiguity'
        stream += event(delta=delta)
        if encoded[12:]:
            stream += event(delta=dict(tool_calls=[dict(index=0,function=dict(arguments=encoded[12:]))]))
        if second_call:
            stream += event(delta=dict(tool_calls=[dict(index=1,id='other',type='function',
                function=dict(name=d.PROPOSE, arguments=d.canonical(ARGS).decode()))]))
    elif finish == 'stop':
        stream += event(delta=dict(content='I would fix the clamp.'))
    usage = dict(prompt_tokens=n, completion_tokens=m, total_tokens=n+m,
                 completion_tokens_details=dict(reasoning_tokens=reasoning)) if metered else None
    stream += event(end=finish, meter=None if usage_only_chunk else usage)
    if usage_only_chunk:
        stream += event(meter=usage, standalone=True)
    return stream + (DONE if end_marker else b'')


class Rehearsal(unittest.TestCase):
    def run_replay(self, replies, counter=None, bundle=None):
        up = d.MockUpstream(replies)
        result = d.replay(bundle or BUNDLE, counter or COUNTER, up)
        self.assertEqual(result['mock_calls'], len(up.dispatched))
        self.assertEqual(result['paid_calls'], 0)
        self.assertEqual(result['gpu_calls'], 0)
        for dim in ('input', 'output'):
            self.assertEqual(result['charged_' + dim], sum(r['charged_' + dim] for r in result['rows']))
        return result

    def test_frozen_fixture_prefix_and_factor_isolation(self):
        f = BUNDLE.files
        short, long = (d.decode(f[f'read-{k}.json']) for k in ('short','long'))
        self.assertEqual(len(f['read-long.json']), 38532)
        self.assertLessEqual(len(f['read-short.json']), 8192)
        self.assertEqual(short['next_offset'], len(short['content'].encode()))
        self.assertTrue(short['truncated'])
        self.assertFalse(long['truncated'])
        self.assertTrue(long['content'].startswith(short['content']))
        self.assertTrue(all(line.startswith('#') for line in long['content'][short['next_offset']:].splitlines()))
        self.assertEqual(long['content'].encode(), f['toy-source.txt'])
        self.assertEqual(short['version'], d.digest(f['toy-source.txt']))
        for effort in ('high','max'):
            a,b = (d.decode(f[f'{effort}-{k}.request.json']) for k in ('short','long'))
            for obj,k in ((a,'short'),(b,'long')):
                self.assertEqual(obj['messages'][2:], d.decode(f[f'history-{k}.json']))
                self.assertEqual(obj['messages'][2]['tool_calls'][0]['id'], obj['messages'][3]['tool_call_id'])
                self.assertTrue(obj['messages'][2]['reasoning_content'])
                self.assertEqual(obj['messages'][3]['content'].encode(), f[f'read-{k}.json'])
            a['messages'][3]['content'] = b['messages'][3]['content']
            self.assertEqual(a,b)
        for k in ('short','long'):
            a,b = (d.decode(f[f'{effort}-{k}.request.json']) for effort in ('high','max'))
            a['reasoning_effort'] = 'max'
            self.assertEqual(a,b)

    def test_four_cells_in_order_with_three_classifications_and_single_success(self):
        result = self.run_replay([response('stop'),
                                  response('length',args='{"express',m=4000,reasoning=3990),
                                  response('tool_calls',args=ARGS,usage_only_chunk=True),
                                  response('tool_calls',args=dict(ARGS,expression='min(100, value)'))])
        rows = result['rows']
        self.assertEqual([r['cell'] for r in rows], list(d.ORDER))
        self.assertEqual(result['dispatched_request_sha256'], [d.digest(BUNDLE.files[n + '.request.json']) for n in d.ORDER])
        self.assertEqual([r['category'] for r in rows], [d.CATEGORIES[i] for i in (1,2,0,1)])
        self.assertEqual([r['detail'] for r in rows], ['no_tool_attempt','generation_truncated','toy_gate_passed','invalid_tool_attempt'])
        self.assertEqual(result['decision'], dict(outcome='eligible_for_separately_authorized_replication', eligible_cell='max-short'))
        self.assertEqual(rows[1]['prefix_state'], 'right_censored_at_response_end')
        self.assertEqual(rows[2]['prefix_state'], 'before_effective_action')
        self.assertEqual(rows[2]['prefix_observed_utf8_bytes'], len(b'synthetic reasoning prefix'))
        self.assertNotEqual(rows[2]['prefix_public_tokenizer_tokens'], rows[2]['reported_usage']['reasoning'])
        self.assertEqual(result['allocated_input'], BUNDLE.manifest['plan']['frozen_input_reservations'])
        self.assertEqual(result['allocated_output'], 16000)
        self.assertEqual((result['charged_input'], result['charged_output']), (400, 4150))
        self.assertNotIn('synthetic reasoning', json.dumps(result))
        self.assertNotIn(ARGS['expression'], json.dumps(result))
        CASES['four_cells'] = result

    def test_all_truncated_stop_screen_and_do_not_count_even_complete_tool(self):
        good_but_truncated = response('length', args=ARGS, m=4000, reasoning=3990)
        reasoning_only = response('length', m=4000, reasoning=4000)
        result = self.run_replay([good_but_truncated, reasoning_only]*2)
        self.assertEqual(result['decision']['outcome'], 'stop_parameter_screen_shrink_task')
        self.assertTrue(all(r['category'] == 'truncated_no_action' for r in result['rows']))
        CASES['all_truncated'] = result

    def test_zero_and_multiple_success_are_not_a_selection(self):
        for pair in ([response('stop')]*2, [response('tool_calls',args=ARGS)]*2):
            result = self.run_replay([*pair, response('stop'), response('length')])
            self.assertEqual(result['decision']['outcome'], 'cannot_distinguish_keep_configuration')
            self.assertIsNone(result['decision']['eligible_cell'])

    def test_strict_tool_schema_target_multiplicity_and_toy_gate(self):
        cases = [response('tool_calls',args=ARGS, second_call=True),
                 response('tool_calls',args=ARGS, wrong_tool=True),
                 response('tool_calls',args=dict(ARGS, path='other.py')),
                 response('tool_calls',args=dict(ARGS, function='other')),
                 response('tool_calls',args=dict(ARGS, extra='not in schema')),
                 response('tool_calls',args=dict(ARGS, expression='print(value)')),
                 response('tool_calls',args='{"path":"toy.py","path":"toy.py"}'),
                 response('stop',args=ARGS)]
        for start in (0,4):
            with self.subTest(start=start):
                result = self.run_replay(cases[start:start+4])
                self.assertTrue(all(r['category'] == 'normal_end_no_action' for r in result['rows']))

    def test_unavailable_prefix_does_not_invent_local_token_count(self):
        result = self.run_replay([response('tool_calls',args=ARGS,ambiguous=True)] + [response('stop')]*3)
        row = result['rows'][0]
        self.assertEqual(row['category'], 'effective_tool_action')
        self.assertEqual(row['prefix_state'], 'boundary_unavailable')
        self.assertIsNone(row['prefix_public_tokenizer_tokens'])
        self.assertEqual(row['reported_usage']['reasoning'], 40)

    def test_usage_over_reservation_stops_and_charges_max_once(self):
        bound = BUNDLE.manifest['cells'][1]['reservation']['local_candidate_bound']
        result = self.run_replay([response('stop'), response('tool_calls',args=ARGS,n=bound+7), response(), response()])
        self.assertEqual((result['mock_calls'], result['stop']), (2,'usage_uncertain'))
        self.assertEqual(result['decision']['outcome'], 'incomplete_evidence')
        row = result['rows'][-1]
        self.assertEqual(row['reported_usage'], dict(input=bound+7,output=50,reasoning=40))
        self.assertEqual((row['charged_input'],row['charged_output'],row['input_overrun']), (bound+7,4000,7))
        self.assertNotIn('category', row)
        CASES['input_overrun'] = result
        out = self.run_replay([response('length',n=50,m=4013,reasoning=4000), response()])
        self.assertEqual((out['mock_calls'],out['charged_output'],out['rows'][0]['output_overrun']), (1,4013,13))
        self.assertEqual(out['stop'], 'usage_uncertain')
        CASES['output_overrun'] = out

    def test_missing_duplicate_and_invalid_usage_fail_closed(self):
        duplicate = response()[:-len(DONE)] + packet(end='stop',meter=dict(prompt_tokens=100,completion_tokens=50,total_tokens=150)) + DONE
        for reply in (response(metered=False), response(reasoning=51), duplicate, b'not SSE'):
            result = self.run_replay([reply, response()])
            self.assertEqual((result['mock_calls'],result['stop']), (1,'usage_uncertain'))
            row = result['rows'][0]
            self.assertIsNone(row['reported_usage'])
            self.assertEqual(row['charged_input'], row['input_reserved'])
            self.assertEqual(row['charged_output'], 4000)
        CASES['missing_usage'] = self.run_replay([response(metered=False), response()])

    def test_fingerprint_variants_stop_without_erasing_valid_usage(self):
        for reply in (response(fp='different'), response(fp=None), response(model='wrong'),
                      packet(delta=dict(role='assistant'), fp='wrong') + response()):
            result = self.run_replay([response('tool_calls',args=ARGS), reply, response(), response()])
            self.assertEqual((result['mock_calls'], result['stop']), (2,'fingerprint_mismatch'))
            self.assertEqual(result['decision']['outcome'], 'incomplete_evidence')
            self.assertEqual((result['charged_input'],result['charged_output']), (200,100))
            self.assertFalse(result['rows'][1]['fingerprint_matches'])
        CASES['fingerprint_mismatch'] = result

    def test_reservation_fail_before_dispatch_uses_zero_for_blocked_cell(self):
        class BadSecond:
            tokenizer = COUNTER.tokenizer

            def reservation(self, req):
                if req['reasoning_effort'] == 'max':
                    return dict(rendered_tokens=14000,margin_tokens=1400,local_candidate_bound=15400)
                return COUNTER.reservation(req)
        result = self.run_replay([response(), response()], counter=BadSecond())
        self.assertEqual((result['mock_calls'], result['stop']), (1,'reservation_failed'))
        self.assertEqual(result['decision']['outcome'], 'incomplete_evidence')
        self.assertFalse(result['rows'][1]['dispatched'])
        self.assertEqual(result['rows'][1]['charged_input'], 0)
        self.assertEqual(result['allocated_output'], 4000)
        CASES['reservation_failed'] = result

    def test_total_preapproval_caps_and_no_fallback(self):
        p = dict(rendered_tokens=500, margin_tokens=512, local_candidate_bound=1012)
        self.assertEqual(d.validate_budget(p, 58988, 12000), 1012)
        for limits in ((58989,12000), (58988,12001)):
            with self.assertRaisesRegex(d.ProtocolError, 'reservation_failed'):
                d.validate_budget(p, *limits)
        class Broken:
            def reservation(self,_):
                raise ValueError('mock counting failure')
        result = self.run_replay([response()], counter=Broken())
        self.assertEqual((result['mock_calls'],result['stop'],result['charged_input']), (0,'reservation_failed',0))

    def test_incomplete_or_unexpected_terminal_does_not_count_as_normal(self):
        reply = response('tool_calls',args=ARGS,end_marker=False)
        result = self.run_replay([reply, response()])
        self.assertEqual((result['mock_calls'], result['stop']), (1,'protocol_invalid'))
        self.assertEqual((result['charged_input'], result['charged_output']), (100,50))
        result = self.run_replay([response('content_filter'), response()])
        self.assertEqual((result['mock_calls'], result['stop']), (1,'classification_unavailable'))

    def test_bad_manifest_or_wire_rejects_before_any_dispatch(self):
        with self.assertRaisesRegex(d.ProtocolError, 'manifest_sha256_mismatch'):
            d.verify(ASSETS, '0'*64)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for path in d.FROZEN.iterdir():
                shutil.copyfile(path,folder/path.name)
            (folder / 'task.txt').write_bytes(b'tampered')
            with self.assertRaisesRegex(d.ProtocolError,'frozen_file_hash_mismatch:task.txt'):
                d.verify(ASSETS, BUNDLE.manifest_sha256, folder)
        last = d.ORDER[-1] + '.request.json'
        corrupt = replace(BUNDLE, files=dict(BUNDLE.files, **{last:b'invalid'}))
        up = d.MockUpstream([response()]*4)
        with self.assertRaisesRegex(d.ProtocolError,'request_sha256_mismatch'):
            d.replay(corrupt, COUNTER, up)
        self.assertEqual(up.dispatched, [])


CASES = {}
if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets', required=True, type=Path)
    cli.add_argument('--expect-sha256', required=True)
    cli.add_argument('--receipt', type=Path)
    opts = cli.parse_args()
    ASSETS = opts.assets
    BUNDLE, COUNTER = d.verify(ASSETS,opts.expect_sha256)
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Rehearsal))
    if not tests.wasSuccessful():
        raise SystemExit(1)
    if opts.receipt:
        with opts.receipt.open('xb') as f:
            f.write(d.manifest_bytes(dict(result='pass_offline_only', protocol=d.ID, manifest_sha256=BUNDLE.manifest_sha256,
                 tests=tests.testsRun, paid_calls=0, gpu_calls=0, cases=CASES)))
