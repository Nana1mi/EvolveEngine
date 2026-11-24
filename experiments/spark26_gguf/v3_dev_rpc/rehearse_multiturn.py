"""Model-free coverage negatives and real Pi offline rehearsals for the registered session."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import p3_multiturn as m

p3 = m.p3
archive = Path(os.environ['LOCALAPPDATA'])/'EvolveEngine'
source = (archive/'v3-pinned-sglang-source/gguf.py').read_bytes()
counter = m.narrow.PublicPromptCounter(archive/'deepseek-flash-1017-08c8657f-offline')


def pages_for_unit():
    # Make distinct contiguous fragments, counted by bytes. Production pages come from Pi.
    out = []
    for start in range(0, len(source), 3000):
        end = min(start + 3000, len(source))
        while end < len(source) and source[end] & 0xc0 == 0x80:
            end += 1
        cid = f'toy-{start}'
        call = dict(name='read',arguments=dict(path=m.FILE,offset=start))
        page = dict(offset=start,content=source[start:end].decode(),version='a'*64,
                    truncated=end < len(source),next_offset=end if end < len(source) else None)
        out.append((cid, call, page))
    return out


def project(items, same_round=False):
    if same_round:
        rounds = [{cid:call for cid,call,_ in items}]
    else:
        rounds = [{cid:call} for cid,call,_ in items]
    history = [dict(role='assistant',tool_calls=[dict(id=cid) for cid in group]) for group in rounds]
    history += [dict(role='tool',tool_call_id=cid,content=p3.canonical(page).decode()) for cid,_,page in items]
    return m.Coverage(source).inspect(history, rounds)


def unit():
    pages = pages_for_unit()
    full = project(pages)
    assert full['met'] and full['source_bytes'] == len(source), full
    dup = [('duplicate', *pages[0][1:])] + pages[:-1]
    assert project(dup)['source_bytes'] < len(source) and not project(dup)['met']
    assert not project(pages[1:])['met']
    assert project(pages, same_round=True)['source_bytes'] == len(source) and not project(pages, same_round=True)['met']
    for bad in ('wrong_path','different_bytes','wrong_version','wrong_next_offset'):
        cid,call,page = pages[0]
        call = dict(call,arguments=call['arguments'].copy())
        page = page.copy()
        if bad == 'wrong_path': call['arguments']['path'] = 'wrong.py'
        if bad == 'different_bytes': page['content'] = 'x'+page['content'][1:]
        if bad == 'wrong_version': call['arguments']['expected_version'] = 'b'*64
        if bad == 'wrong_next_offset': page['next_offset'] = 14
        assert not project([(cid,call,page),*pages[1:]])['met'], bad
    assert not m.summary(dict(first_qualified_turn=3,rpc=[]))['accepted_delivery']
    original = m.narrow.NarrowContract(source)
    inside = source.replace(b'        shard_id = layer.qweight.shard_id\n\n        if shard_id:',
                            b'        shard_id = layer.qweight.shard_id  # offline fixture\n\n        if shard_id:',1)
    original.from_args({'files':{m.FILE:inside.decode()}})
    try:
        original.from_args({'files':{m.FILE:(inside+b'\n# out of scope\n').decode()}})
    except ValueError:
        pass
    else:
        raise AssertionError('outside function edit accepted')


def rehearsal(output, mode):
    requests = []
    scenes = p3.record(p3.HERE/'g1a-03.json')['common_runner']['worker'][1]['result']['scenes']
    action_index = 0
    read_count = 0

    @contextmanager
    def send(payload, _timeout):
        nonlocal action_index, read_count
        req = json.loads(payload)
        requests.append(req)
        turn = len(requests)
        assert req['max_tokens'] == 8192 and req['reasoning_effort'] == 'max' and req['thinking']=={'type':'enabled'}
        assert {'ee_rpc','read','bash'} <= {t['function']['name'] for t in req['tools']}
        if turn > 1:
            assert req['messages'][-1]['role']=='tool'
        action = None
        finish = 'tool_calls'
        if turn == 1:
            action = ('read', dict(path=m.FILE,offset=0))
        elif mode == 'early_stop':
            finish = 'stop'
        elif action_index == 0:
            page = json.loads(req['messages'][-1]['content'])
            read_count += 1
            if page['truncated']:
                action = ('read', dict(path=m.FILE,offset=page['next_offset'],expected_version=page['version']))
            elif mode == 'length':
                finish = 'length'
            else:
                action = ('bash', dict(command=f"sed -i '512s/$/  # offline fixture/' {m.FILE}"))
                action_index += 1
        elif action_index in (1,2):
            op = 'develop_quality' if action_index == 1 else 'deliver_engine'
            action = ('ee_rpc', dict(operation=op,arguments=dict(paths=[m.FILE])))
            action_index += 1
        else:
            finish = 'stop'

        def event(delta=None, end=None, usage=None):
            return b'data: '+p3.canonical(dict(id=f'fixture-{turn}',model='deepseek-flash',system_fingerprint=m.FINGERPRINT,
                   choices=[dict(index=0,delta=delta or {},finish_reason=end)],usage=usage))+b'\n\n'
        stream = event(dict(role='assistant',reasoning_content=f'fixture reasoning {turn}'))
        if action:
            name, args = action
            stream += event(dict(tool_calls=[dict(index=0,id=f'call_mock_{turn}',type='function',
                                                   function=dict(name=name,arguments=p3.canonical(args).decode()))]))
        if finish=='length':
            stream += event(dict(tool_calls=[dict(index=0,id=f'call_mock_{turn}',type='function',
                                  function=dict(name='bash',arguments='{"command":"echo partial'))]))
        elif finish=='stop':
            stream += event(dict(content='fixture complete'))
        produced = 8192 if finish=='length' else 50
        stream += event(end=finish,usage=dict(prompt_tokens=500,completion_tokens=produced,total_tokens=500+produced,
                                      completion_tokens_details=dict(reasoning_tokens=produced if finish=='length' else 5)))
        yield (stream+b'data: [DONE]\n\n').splitlines(keepends=True)

    def worker(op,edits,engine,_base,_lease,job):
        if op == 'develop_measure':
            assert edits == {}
            status, data = 'observed', {'scenes':scenes}
        else:
            assert mode == 'candidate' and list(edits)==[m.FILE] and b'# offline fixture' in edits[m.FILE]
            status, data = 'valid', {}
        receipt = p3.DevReceipt(engine,op,status,100,p3.digest(op.encode()),data)
        job.update(status=status,receipt_id=receipt.receipt_id,node_ms=100)
        return receipt

    result = m.run(source,output,counter,send,worker)
    assert len(result['provider'])==len(requests) and all(t['usage']=='settled' for t in result['provider']), result
    if mode == 'early_stop':
        assert len(requests)==2 and result['first_qualified_turn'] is None and result['status']=='session_completed'
        assert not result['acceptance']['accepted_delivery']
    else:
        gate = result['first_qualified_turn']
        assert gate==read_count+1 and gate>=4 and result['provider'][gate-1]['context']['source_bytes']==m.SOURCE_BYTES
        if mode == 'length':
            assert gate==len(requests) and result['stop_cause']=='generation_truncated'
            assert len(result['runner']['pi_tool_executions'])==read_count
            assert result['provider'][-1]['reasoning_tokens']==8192 and not result['workers']
        else:
            assert result['status']=='session_completed' and result['acceptance']['accepted_delivery']
            assert result['development_calls']==1 and result['provider'][gate]['tool_call_prefixes'][0]['operation']=='develop_quality'
    return dict(mode=mode,success=True,model_calls=len(requests),context_turn=result['first_qualified_turn'],
                stop=result['stop_cause'],accepted_delivery=result['acceptance']['accepted_delivery'])


if __name__ == '__main__':
    unit()
    with tempfile.TemporaryDirectory() as root:
        reports = [rehearsal(Path(root)/mode,mode) for mode in ('early_stop','length','candidate')]
    print(json.dumps(dict(status='passed_offline',cases=reports),sort_keys=True), flush=True)
