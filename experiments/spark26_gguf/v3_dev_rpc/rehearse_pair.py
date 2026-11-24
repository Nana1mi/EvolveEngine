"""Rehearse exactly the launcher's S1→S2 handoff with real Pi and mock transports."""
from contextlib import contextmanager
from inspect import getsource
import json
from pathlib import Path
import sys
import tempfile

from p3 import ALLOWED, SMOKE, SYSTEM, EMPTY, PI, DevReceipt, Snapshot, canonical, digest, need, execution_protocol, pilot_profile, provider_diagnostics, record, run_pair, runtime_for
from verify_common import chunks

EXT='extensions/handoff.mjs'
CODE='''export default function(pi) {
  pi.registerTool({name:'handoff_witness',label:'Handoff witness',description:'Synthetic fixture',
    parameters:{type:'object',properties:{}},
    async execute(){return {content:[{type:'text',text:'ACTUAL_H1_CALL'}],details:{}};}
  });
}
'''
BASE=b'def value():\n    return 1\n'
E1='def value():\n    return 731\n'
PRIVATE='not_carried_y4w6'


def run(result):
    profile=pilot_profile();runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),profile,SYSTEM,None)
    load=dict(EMPTY,extensions=[EXT]);h0=Snapshot.make({},EMPTY,runtime);h1=Snapshot.make({EXT:CODE.encode()},load,runtime)
    src=ALLOWED[0];seen={'S1':0,'S2':0};observations={};workers=[]
    def rpc(operation,**arguments):return 'ee_rpc',dict(operation=operation,arguments=arguments)
    plans={'S1':[
        ('write',dict(path=src,content=E1)),rpc('develop_quality',paths=[src]),rpc('deliver_engine',paths=[src]),
        ('write',dict(path='../harness/'+EXT,content=CODE)),rpc('harness_submit',paths=[EXT],load=load),
        rpc('harness_activate',version=h1.identity,expected=h0.identity),
        ('write',dict(path=src,content='def value():\n    return 999\n')),
        ('write',dict(path='../'+PRIVATE,content='session-1-only'))],
       'S2':[
        ('read',dict(path=src)),rpc('facts'),('bash',dict(command=f'test ! -e /scratch/{PRIVATE} && printf PRIVATE_ABSENT')),
        ('handoff_witness',{}),rpc('develop_quality',paths=[])]}
    expectations={'S1':{3:'valid',4:'signed',6:h1.identity,7:h1.identity},
                  'S2':{2:'return 731',3:digest(b'fixture-receipt-1'),4:'PRIVATE_ABSENT',5:'ACTUAL_H1_CALL',6:'valid'}}

    def model_for(arm):
        @contextmanager
        def reply(wire,_left):
            j=json.loads(wire);seen[arm]+=1;turn=seen[arm];tools={x['function']['name'] for x in j['tools']}
            need(('handoff_witness' in tools)==(arm=='S2') and {'read','write','bash','ee_rpc'}<=tools,'wrong extension exposure')
            values=[x['content'] for x in j['messages'] if x['role']=='tool']
            if turn in expectations[arm]:
                witness=expectations[arm][turn];need(values and witness in json.dumps(values[-1]),f'missing scoped tool result {arm}:{turn}')
                observations[f'{arm}:{turn}']=witness
            if turn==1:need(not values and len(j['messages'])==2,'a previous transcript was injected')
            if arm=='S2':need(not any('return 999' in json.dumps(val) or 'session-1-only' in json.dumps(val) for val in values),'unregistered data carried over')
            yield chunks(turn,plans[arm][turn-1] if turn<=len(plans[arm]) else None).splitlines(keepends=True)
        return reply

    def worker(op,edits,engine,base,lease,job):
        if job.get('collocated_e0'):
            need(op=='develop_measure' and not edits and base=={src:BASE},'calibration is not E0')
            return DevReceipt(engine,op,'observed',50,digest(b'calibration-fixture'),{'scenes':{k:{'score':1.0} for k in ('W1','W2','W3','W4')}})
        need(op=='develop_quality' and edits=={src:E1.encode()} and base=={src:BASE} and lease==200000,'wrong exact artifact composition at worker')
        workers.append(dict(engine=engine,source=digest(edits[src])));data=dict(fixture=True)
        return DevReceipt(engine,op,'valid',20,digest(f'fixture-receipt-{len(workers)}'.encode()),data)

    def replay(job):need(job['receipt_id'] in {digest(x) for x in (b'calibration-fixture',b'fixture-receipt-1')},'unknown fixture receipt')

    with tempfile.TemporaryDirectory(prefix='ee-v3-pair-fixture-') as directory:
        dest=Path(directory)/'run';run_pair(BASE,dest,model_for,worker,replay)
        s1=record(dest/'S1.json');s2=record(dest/'S2.json');result.update(runtime=runtime,orchestration_sha=digest(getsource(run_pair).encode()),protocol_sha=digest(Path(__file__).with_name('p3_protocol.py').read_bytes()),handoff=record(dest/'S1-to-S2.json'),S1=s1,S2=s2,observed=observations)
        result.update(provider_diagnostics=provider_diagnostics(),execution_protocol=execution_protocol())
        status=record(dest/'P3-status.json')
        need(all(status[k]==result[k] for k in ('provider_diagnostics','execution_protocol')),'missing protocol binding in lifecycle')
        for s in (s1,s2):
            need(not s['provider_rejections'] and all(r['request_meta']['profile']=='pinned_max' and
                 r['request_meta']['replay']['reasoning_mismatches']==0 and r['request_meta']['request_bytes']>0 and
                 r['request_meta']['upstream_request_bytes']>0 for r in s['provider']), 'broker request projection incomplete')
            need(any(r['request_meta']['tool_result_count']>0 for r in s['provider']), 'tool result sizes not observed')
        need(s1['end_E']==s2['start_E'] and s1['start_E']!=s1['end_E'] and s1['next_H']==s2['start_H']==h1.identity,'E/H handoff did not use explicit selection')
        need(s1['runner']['run']!=s2['runner']['run'] and s2['runner']['snapshot']==h1.identity and all(s['runner']['isolated_state_removed'] for s in (s1,s2)),'session fork not cleared')
        need(s1['explicitly_delivered'] and not s2['explicitly_delivered'] and len(workers)==2 and s2['ingress_facts_sha']!=digest(canonical({})),'wrong authorized ingress')
        need((dest/'trusted'/'S2'/src).read_bytes()==E1.encode(),'engine on disk drifted')
        need(result['handoff']['status']=='validated' and result['handoff']['facts_sha']==s2['ingress_facts_sha'] and s1['node_ms']==s2['node_ms']==70,'calibration or handoff missing from receipt')
    result['status']='pass_pair_launcher_fixture'


if __name__=='__main__':
    j=dict(status='not_passed',scope='actual P3 launcher handoff with scripted Pi transport and mock quality; no GPU or paid calls')
    with Path(sys.argv[1]).open('x',encoding='utf8') as f:
        try:run(j)
        except (ValueError,OSError,RuntimeError,TypeError,KeyError) as e:j['error']=str(e)
        finally:f.write(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    print(j['status'],j.get('error',''));raise SystemExit(j['status']=='not_passed')
