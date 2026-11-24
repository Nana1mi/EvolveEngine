"""Exercise the 384K path with real Pi and local model and GPU fixtures."""
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from itertools import chain, repeat
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

import p3_explore as explore

p3 = explore.p3
archive = Path(os.environ['LOCALAPPDATA']) / 'EvolveEngine'
FIXTURE = b'  # exploration offline fixture'
UNQUALIFIED_EDIT = b'  # unqualified edit after the quality receipt'
CAP = 393216
REASON = 'synthetic long response for tool replay 0123456789 ' * 42000
MARKER = 'Offline Harness checkpoint is active.'


def case(root, mode, source, counter):
    config, limits = explore.load_config(explore.CONFIG)
    if mode == 'budget':
        limits = replace(limits, output=CAP)
        config['budget']['output_tokens'] = limits.output
    if mode == 'tail':
        limits = replace(limits, output=CAP+2)
        config['budget']['output_tokens'] = limits.output
    if mode == 'cost':
        config['budget']['api_cost_rmb'] = '3.16'
    if mode == 'context':
        config['transport']['relay_socket_idle_seconds'] = 1
        class ContextFixture:
            def reservation(self, request):
                result = counter.reservation(request)
                if request['messages'][-1]['role'] == 'tool':
                    return dict(result, local_candidate_bound=1_000_000-CAP+1)
                return result
        active_counter = ContextFixture()
    else:
        active_counter = counter
    outbound = []
    fixture_errors = []
    scenes = p3.record(p3.HERE/'g1a-03.json')['common_runner']['worker'][1]['result']['scenes']
    file = explore.narrow.FILE
    paths = dict(paths=[file])
    read = ('read', dict(path=file, offset=0))
    engine_edit = ('bash', dict(command=f"sed -i '512s/$/{FIXTURE.decode()}/' {file}"))

    def rpc(operation, arguments):
        return 'ee_rpc', dict(operation=operation, arguments=arguments)

    if mode == 'delivery':
        steps = [read, engine_edit, rpc('develop_quality', paths), rpc('develop_measure', paths),
                 rpc('deliver_engine', paths)]
    elif mode == 'checkpoint':
        dirty_and_stage_harness = (
            f"sed -i '512s/$/{UNQUALIFIED_EDIT.decode()}/' {file} && "
            f"grep -Fq -- '{UNQUALIFIED_EDIT.decode()}' {file} && "
            'mkdir -p /scratch/harness/prompts && '
            f"printf '%s' '{MARKER}' > /scratch/harness/prompts/recovery.md && touch transient.txt"
        )
        verify_restored = (
            f'test ! -e transient.txt && grep -Fq -- "{FIXTURE.decode()}" {file} && '
            f'! grep -Fq -- "{UNQUALIFIED_EDIT.decode()}" {file}'
        )
        steps = [engine_edit, rpc('develop_quality', paths),
                 ('bash', dict(command=dirty_and_stage_harness)),
                 rpc('harness_submit', dict(paths=['prompts/recovery.md'], load=dict(p3.EMPTY, prompts=['prompts/recovery.md']))),
                 'activate', 'length', ('bash', dict(command=verify_restored)),
                 rpc('develop_measure', paths), rpc('deliver_engine', paths)]
    elif mode == 'length':
        steps = ['length', 'length']
    elif mode in ('budget', 'cost'):
        steps = ['length']
    elif mode == 'context':
        steps = [read, None]
    elif mode == 'tail':
        steps = [read]
    else:
        raise AssertionError(mode)

    @contextmanager
    def fixture(payload, _timeout):
        request = json.loads(payload)
        turn = len(outbound)
        assert turn < len(steps), 'unexpected upstream call after stop'
        step = steps[turn]
        if mode == 'context' and turn == 0:
            sleep(1.25)
        outbound.append(dict(size=len(payload), cap=request['max_tokens']))
        assert request['thinking'] == {'type':'enabled'} and request['reasoning_effort'] == 'max'
        assert request['stream'] and request['stream_options']['include_usage']
        assert request['max_tokens'] == CAP
        if turn == 1 and mode == 'delivery':
            assert len(payload) > 2*1024*1024
            assistant = [m for m in request['messages'] if m['role'] == 'assistant']
            assert assistant[0]['reasoning_content'] == REASON
            page = json.loads(request['messages'][-1]['content'])
            assert page['truncated'] and 0 < page['next_offset'] <= 8192
        if step == 'activate':
            h1 = json.loads(json.loads(request['messages'][-1]['content'])['result'])['version']
            h0 = p3.record(root/'registration.json')['start_H']
            step = rpc('harness_activate', dict(version=h1, expected=h0))
        if (mode == 'checkpoint' and turn == 6) or (mode in ('length','context') and turn == 1):
            assert [m['role'] for m in request['messages']] == ['system', 'user']
            user = request['messages'][1]['content']
            if isinstance(user, list):
                user = '\n'.join(part.get('text', '') for part in user)
            if mode == 'checkpoint':
                assert MARKER in request['messages'][0]['content']
                assert 'valid develop_quality receipt' in user
            else:
                assert 'current file is the original E0' in user

        def event(delta=None, finish=None, usage=None):
            return b'data: '+p3.canonical(dict(id=f'fixture-{turn}', model='deepseek-flash',
                system_fingerprint=explore.narrow.FINGERPRINT,
                choices=[dict(index=0,delta=delta or {},finish_reason=finish)], usage=usage))+b'\n\n'

        parts = [event(dict(role='assistant'))]
        if turn == 0 and mode == 'delivery':
            parts.extend(event(dict(reasoning_content=REASON[i:i+8192])) for i in range(0,len(REASON),8192))
        truncated = step == 'length'
        if step is not None:
            tool, args = read if truncated else step
            arguments = '{"path":"unfinished' if truncated else p3.canonical(args).decode()
            parts.append(event(dict(tool_calls=[dict(index=0,id=f'call_{turn}',type='function',
                                          function=dict(name=tool,arguments=arguments))])))
        else:
            parts.append(event(dict(content='Finished.')))
        generated = CAP if truncated or (turn == 0 and mode == 'delivery') else 3
        parts.extend([event(finish='length' if truncated else 'stop' if step is None else 'tool_calls', usage=dict(
            prompt_tokens=500,completion_tokens=generated,total_tokens=500+generated,
            completion_tokens_details=dict(reasoning_tokens=generated if truncated else generated-1),
            prompt_cache_hit_tokens=200,prompt_cache_miss_tokens=300,prompt_tokens_details=dict(cached_tokens=200))),
            b'data: [DONE]\n\n'])
        comment = b':'+b'x'*2045+b'\n\n'
        frames = (193*1024*1024-sum(map(len,parts)))//len(comment) if mode == 'delivery' and turn == 0 else 0
        outbound[-1]['sse_bytes'] = sum(map(len,parts)) + frames*len(comment)
        yield chain(repeat(comment, frames), parts)

    @contextmanager
    def send(payload, timeout):
        try:
            with fixture(payload, timeout) as stream:
                yield stream
        except (AssertionError, KeyError, TypeError) as exc:
            fixture_errors.append(f'{type(exc).__name__} at upstream call {len(outbound)}: {exc}')
            raise ValueError('offline transport fixture failed') from exc

    def worker(op, edits, engine, _base, lease, job):
        assert lease == 360_000
        status = 'observed' if op == 'develop_measure' else 'valid'
        if edits:
            assert FIXTURE in edits[file]
        data = dict(scenes=scenes) if status == 'observed' else {}
        receipt = p3.DevReceipt(engine,op,status,100,p3.digest(op.encode()),data)
        job.update(operation=op,status=status,node_ms=100,receipt_id=receipt.receipt_id)
        return receipt

    result = explore.run(source, root, config, limits, active_counter, send, worker)
    assert not fixture_errors, fixture_errors
    assert len(outbound) == len(steps), result
    assert not result['runner']['transport_fault'], result
    assert result['runner']['relay_model_socket_idle_seconds'] == config['transport']['relay_socket_idle_seconds']
    if mode in ('delivery', 'checkpoint'):
        assert result['status'] == 'agent_delivered' and result['stop_cause'] == 'agent_delivery', result
        assert result['runner']['gateway_events'][-1]['reason'] == 'agent_delivered'
        assert result['workers'][-1]['performance_dev']['index'] == 1.0
        assert result['delivered'] and result['development_calls'] == 2 and result['node_ms'] == 300
    if mode == 'checkpoint':
        first, second = result['segments']
        recovery = result['recovery']
        assert first['stop_cause'] == 'generation_truncated' and second['start_H'] != first['start_H']
        assert first['runner']['pi_tool_executions'][-1]['name'] == 'ee_rpc'
        assert all(not x['error'] for x in first['runner']['pi_tool_executions'])
        assert all(not x['error'] for x in second['runner']['pi_tool_executions'])
        assert second['runner']['provider_rows'][0]['prompt_fragments'] == [p3.digest(MARKER.encode())]
        assert second['limits']['input'] == limits.input - 6*500
        assert second['limits']['output'] == limits.output - CAP - 15
        assert second['limits']['turns'] == limits.turns - 6
        assert second['limits']['wall_ms'] < first['limits']['wall_ms']
        assert result['workers'][-1]['api_output_before'] == CAP+21
        assert recovery['quality_receipt'] == result['workers'][0]['receipt_id']
        restored = (root/'recovery-gguf.py').read_bytes()
        assert FIXTURE in restored and UNQUALIFIED_EDIT not in restored
        assert recovery['source_sha'] == result['delivered_sha256'] == p3.digest(restored)
    elif mode in ('length', 'budget'):
        assert result['stop_cause'] == 'generation_truncated' and not result['workers'] and 'delivered' not in result
        assert all(not seg['runner']['pi_tool_executions'] for seg in result['segments'])
        if mode == 'length':
            assert len(result['segments']) == 2 and result['recovery']['quality_receipt'] is None
        else:
            assert result['recovery_blocked'] == 'remaining_budget' and len(result['segments']) == 1
    elif mode == 'context':
        assert result['stop_cause'] == 'agent_finished' and result['recovery']['trigger'] == 'context_generation_window', result
        assert len(result['segments']) == 2 and result['usage']['input'] == 1000
        assert result['provider'][0]['wall_ms'] >= 1200
    elif mode == 'cost':
        assert result['stop_cause'] == 'api_cost_limit' and len(result['segments']) == 2, result
        assert not result['provider_rejections'][-1]['dispatched']
        assert not (root/'model-attempts-recovery'/'0001-cost-reservation.json').exists()
    elif mode == 'tail':
        assert result['stop_cause'] == 'full_generation_window_unavailable' and len(result['segments']) == 1
    ledger = p3.record(root/'model-attempts'/'0001-result.json')
    assert ledger['usage']['usage'] == 'settled' and ledger['usage']['reported_cache_usage'] == dict(hit=200,miss=300)
    assert ledger['estimated_rmb']['peak']['upper'] == ledger['estimated_rmb']['peak']['lower']
    assert result['usage']['output_including_reasoning'] == sum(item['output_including_reasoning'] for item in result['provider'])
    cost_summary = result['api_cost_budget']
    assert Decimal(cost_summary['total_charged_rmb']) == explore.peak_uncached_rmb(
        result['usage']['input'], result['usage']['output_including_reasoning'])
    assert Decimal(cost_summary['total_charged_rmb']) <= Decimal(cost_summary['limit_rmb'])
    assert Decimal(cost_summary['outstanding_reservation_rmb']) == 0
    assert p3.record(root/'api-cost-budget.json') == cost_summary
    return dict(mode=mode, caps=[item['cap'] for item in outbound], settled_output=result['usage']['output_including_reasoning'],
                max_request_bytes=max(item['size'] for item in outbound),
                max_sse_bytes=max(item['sse_bytes'] for item in outbound), result=result['stop_cause'])


def reservation_rehearsal(root, counter):
    from evolveengine.v3_provider import LeaseError
    config, limits = explore.load_config(explore.CONFIG)
    request = dict(model='deepseek-flash',thinking={'type':'enabled'},reasoning_effort='max',stream=True,
                   max_tokens=limits.per_turn_output,messages=[dict(role='user',content='Test reservation.')])
    expected = counter.reservation(request)['local_candidate_bound'] + 1
    sent = []

    @contextmanager
    def send(payload, _idle):
        sent.append(json.loads(payload))
        reported = dict(prompt_tokens=expected,completion_tokens=2,total_tokens=expected+2)
        chunk = dict(model='deepseek-flash',system_fingerprint=explore.narrow.FINGERPRINT,
                     choices=[dict(index=0,delta={},finish_reason='stop')],usage=reported)
        yield [b'data: '+p3.canonical(chunk)+b'\n\n',b'data: [DONE]\n\n']

    lease = explore.ExplorationLease(limits,send,counter,config['model']['context_window'],root,
                                     api_cost=explore.ApiCostBudget(Decimal('40')), model_socket_idle_seconds=900)
    for _ in range(2):
        try:
            lease.exchange(p3.canonical(request))
        except LeaseError:
            pass
        else:
            raise AssertionError('overrun did not close the lease')
    assert len(sent) == 1
    result = p3.record(root/'0001-result.json')
    assert result['usage']['usage'] == 'usage_uncertain' and result['usage']['input_overrun'] == 1
    assert result['estimated_rmb'] is None and result['cumulative_charge'] == dict(input_tokens=expected,output_tokens=CAP)
    assert Decimal(result['usage']['api_cost']['charged_rmb']) == explore.peak_uncached_rmb(expected, CAP)
    assert p3.record(root/'0002-result.json')['rejection']['reason'] == 'turn_or_poisoned_lease'
    return dict(mode='input_overrun',result=result['usage']['usage'],input_overrun=1,upstream_calls=len(sent))


if __name__ == '__main__':
    source = (archive/'v3-pinned-sglang-source/gguf.py').read_bytes()
    counter = explore.narrow.PublicPromptCounter(archive/'deepseek-flash-1017-08c8657f-offline')
    with TemporaryDirectory() as root:
        outcomes = [case(Path(root)/mode, mode, source, counter) for mode in ('delivery','checkpoint','length','budget','context','tail','cost')]
        outcomes.append(reservation_rehearsal(Path(root)/'overrun',counter))
    print(json.dumps(dict(status='passed_offline',cases=outcomes),sort_keys=True),flush=True)
