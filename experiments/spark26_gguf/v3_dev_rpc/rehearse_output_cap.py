"""Exercise actual pinned Pi, relay, broker and P3 S1 stop, with only local fixtures."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile

from p3 import (ALLOWED, CAP, DEV_CALLS, EMPTY, GENERATION_TRUNCATED, NODE, PI, PROMPT, SMOKE, SYSTEM, TRAIN_MODES,
                DevReceipt, Lineage, Snapshot, SourceContract, canonical, digest, episode_gateway, execution_protocol,
                need, pilot_profile, provider_diagnostics, record, run_pair, runtime_for, sandbox)

SOURCE=b'def fixture():\n    return 1\n'
E0={ALLOWED[0]:SOURCE}
MODEL='deepseek-flash'
FP='offline-cap-fixture'
CAPS=[8192,8192,7616]


def sse(turn,finish,completion,*,action=None,partial=False):
    model=dict(id=f'chatcmpl-output-fixture-{turn}',object='chat.completion.chunk',created=0,model=MODEL,system_fingerprint=FP)
    def evt(delta=None,reason=None,usage=None):
        return b'data: '+canonical(model|dict(choices=[dict(index=0,delta=delta or {},finish_reason=reason)],usage=usage))+b'\n\n'
    data=evt(dict(role='assistant',reasoning_content=f'toy thought {turn}'))
    if action or partial:
        args='{"operation":"facts","arguments":{' if partial else canonical(dict(operation='facts',arguments={})).decode()
        data+=evt(dict(tool_calls=[dict(index=0,id=f'call_fixture_{turn}',type='function',function=dict(name='ee_rpc',arguments=args))]))
    if finish=='stop':data+=evt(dict(content='fixture complete'))
    reasoning=completion if finish=='length' and not partial else completion-10
    usage=dict(prompt_tokens=100+turn,completion_tokens=completion,total_tokens=100+turn+completion,
               completion_tokens_details=dict(reasoning_tokens=reasoning))
    # DeepSeek's final choice chunk may itself carry the usage.
    return (data+evt(reason=finish,usage=usage)+b'data: [DONE]\n\n').splitlines(keepends=True)


def transport_for(scenario,wire_rows):
    @contextmanager
    def send(wire,_timeout):
        request=json.loads(wire);i=len(wire_rows);turn=i+1
        need(i<(3 if scenario=='normal' else 1),'unexpected follow-up request')
        shape={k:request[k] for k in ('model','thinking','reasoning_effort','max_tokens','stream_options')}
        need(shape==dict(model=MODEL,thinking={'type':'enabled'},reasoning_effort='max',max_tokens=CAPS[i],
                         stream_options={'include_usage':True}),'wrong actual outbound Max or output limit')
        tools={x['function']['name'] for x in request['tools']}
        need({'bash','read','write','edit','ee_rpc'}<=tools,'shared tool set unexpectedly narrowed')
        wire_rows.append(dict(shape,tools=sorted(tools)))
        if scenario=='normal':
            if turn>1:need(request['messages'][-1]['role']=='tool' and 'facts' in request['messages'][-1]['content'],
                           'real Pi did not replay the preceding tool result')
            yield sse(turn,'tool_calls' if turn<3 else 'stop',CAPS[i],action=turn<3)
        else:yield sse(turn,'length',8192,partial=scenario=='partial_tool')
    return send


def worker_forbidden(*_):
    raise AssertionError('a DEV worker must not be invoked in the output broker fixture')


def single(root,arm,scenario,runtime,profile):
    wire=[];lineage=Lineage(root,runtime);snapshot=Snapshot.make({},EMPTY,runtime);lineage.initialize(snapshot)
    contract=SourceContract(E0,ALLOWED,max_files=1,max_bytes=160*1024,max_text_growth=1024,max_new_literal=1024)
    gateway,model,policy=episode_gateway(arm,lineage,contract,{},worker_forbidden,transport_for(scenario,wire),
                                       remain=CAP.wall_ms,node_left=NODE-50,fingerprint=FP)
    runner=sandbox(PI,E0,snapshot,runtime,profile,SYSTEM,PROMPT,gateway,seconds=80)
    need(runner['pi']=='1.5.2' and not runner['isolation']['gpu'] and runner['isolated_state_removed'] and runner['socket_removed']
         and runner['unchanged_install'] and not runner['transport_fault'],'fixture runner did not isolate or clean up')
    need(policy.mode==TRAIN_MODES[arm] and policy.calls_left==DEV_CALLS and not model.rejections and not gateway.events,
         'policy or metering diverged')
    need(all(r['usage']=='settled' for r in model.rows),'a response did not settle')
    if scenario=='normal':
        need([x['finish'] for x in model.rows]==['tool_calls','tool_calls','stop'] and model.spent_output==CAP.output and
             runner['status']=='pi_session_complete' and [p['operation'] for p in policy.rows]==['facts','facts'] and
             runner['pi_tool_executions']==[dict(name='ee_rpc',error=False)]*2,'normal Pi loop or budget accounting failed')
        need([x['request_meta']['output_left'] for x in model.rows]==[24000,15808,7616], 'wrong output pre-request remainder')
    else:
        need(len(wire)==len(model.rows)==1 and model.rows[0]['result']==GENERATION_TRUNCATED and model.poisoned and
             model.spent_output==8192 and runner['status']=='not_completed' and not runner['pi_tool_executions'] and
             not policy.rows and not model.seen_ids,'a truncated generation was continued, executed or mischarged')
        if scenario=='reasoning_only':need(model.rows[0]['reasoning_tokens']==8192,'reasoning-only case was not simulated')
    return dict(mode=policy.mode,wire=wire,settled_output=model.spent_output,
                turns=[{k:r[k] for k in ('finish','result','usage','reasoning_tokens','output_including_reasoning','request_meta')} for r in model.rows],
                pi_status=runner['status'],tool_events=runner['pi_tool_executions'],trusted_events=policy.rows,isolated_state_removed=True)


def prove_no_s2(root):
    wire=[];workers=[]
    def model_for(episode):
        need(episode=='S1','S2 must not start after S1 truncation')
        return transport_for('partial_tool',wire)
    def worker(op,edits,engine,base,lease,job):
        need(job.get('collocated_e0') and op=='develop_measure' and not edits and base==E0,'unexpected candidate worker in truncation fixture')
        workers.append(op)
        return DevReceipt(engine,op,'observed',50,digest(b'mock-E0'),dict(scenes={k:dict(score=1) for k in ('W1','W2','W3','W4')}))
    def replay(job):need(job['receipt_id']==digest(b'mock-E0'),'not the fixture E0')
    try:run_pair(SOURCE,root,model_for,worker,replay,fingerprint=FP)
    except RuntimeError as e:need(str(e)=='pilot did not settle; no automatic retry or continuation','unexpected launcher failure')
    else:raise ValueError('launcher treated truncation as a complete session')
    s1=record(root/'S1.json');status=record(root/'P3-status.json');w=record(root/'W-budget-proposal.json')
    need(s1['status']=='unsettled' and s1['failure']==status['stop_class']==GENERATION_TRUNCATED and status['stage']=='S1_agent'
         and not s1['budget_stop'] and not s1['workers'] and not s1['rows'] and s1['explicitly_delivered'] is None
         and s1['usage']['output_including_reasoning']==8192 and workers==['develop_measure'] and len(wire)==1
         and not any((root/x).exists() for x in ('S1-to-S2.json','S2-E0.json','S2.json'))
         and w['status']=='hold_requires_settled_uncensored_costs' and not w['w_started'], 'bad P3 truncation or S2/W status')
    return dict(status=status,s1_status=s1['status'],s1_failure=s1['failure'],s1_usage=s1['usage'],s2_started=False,w=w)


def run(out):
    profile=pilot_profile();runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),profile,SYSTEM,None)
    need(profile['maxTokens']==CAP.per_turn_output==8192 and CAP.output==24000 and CAP.input==200000,'wrong locked caps')
    out.update(runtime=runtime,provider_diagnostics=provider_diagnostics(),execution_protocol=execution_protocol(),
               fixture_sha=digest(Path(__file__).read_bytes()),observations={})
    with tempfile.TemporaryDirectory(prefix='ee-v3-8192-offline-') as tmp:
        base=Path(tmp)
        for scenario in ('normal','reasoning_only','partial_tool'):
            pair={}
            for arm in ('B0','B2'):
                print(arm,scenario,flush=True)
                pair[arm]=single(base/f'{arm}-{scenario}',arm,scenario,runtime,profile)
            comparable=lambda r:{k:r[k] for k in ('wire','settled_output','pi_status','tool_events')}
            need(comparable(pair['B0'])==comparable(pair['B2']),'matched B0/B2 behavior differs')
            out['observations'][scenario]=pair
        out['launcher_truncation']=prove_no_s2(base/'pair-truncated')
    out['status']='pass_offline_max_output'


if __name__=='__main__':
    out=dict(status='not_passed',scope='actual pinned Pi, simulated SSE and CPU-only; no paid calls, no GB10 or free pilot',gpu_calls=0,paid_calls=0)
    with Path(sys.argv[1]).open('x',encoding='utf-8',newline='\n') as f:
        try:run(out)
        except (ValueError,OSError,RuntimeError,KeyError,AssertionError) as e:out['error']=str(e)
        finally:f.write(json.dumps(out,indent=2)+'\n')
    print(out['status'],out.get('error',''));raise SystemExit(out['status']!='pass_offline_max_output')
