"""Same V3 Sandbox/Max/Session path; scripted fixture only, no paid calls or GPU."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import sys
import tempfile

from shared_runner_smoke import IM, BASE, FACT, SYSTEM, PROMPT, EMPTY, verify
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from broker import chunks
from evolveengine.v3_harness import Lineage, Snapshot, canonical, digest
from evolveengine.v3_provider import MaxLease, Limits
from evolveengine.v3_runner import Gateway, runtime_for, sandbox
from evolveengine.v3_session import DevReceipt, Session, SourceContract
from run import judge

SHARED='extensions/fixture_base.mjs'; COUNT='deps/count.mjs'; SHARED_BYTES=(HERE/'fixture_shared.mjs').read_bytes()
LOAD=dict(EMPTY,extensions=[SHARED]); GOOD='export function count(xs){return xs.filter(x=>x==="hit").length;}\n'
BAD='export function count(xs){const n=xs.filter(x=>x==="hit").length;if(!n)throw Error("TOY_ZERO_BUG");return n;}\n'
SOURCE=BASE|{'facts.txt':b'toy prior fact: count exact hits\n'};TOOLS=['read','write','ee_rpc','test_ops','count_*']
MODES={'train2':'evolving_train','train0':'fixed_train','C1':'evaluation','C2':'evaluation','B':'evaluation','stay':'evolving_train','revert':'evolving_train','reopened':'evolving_train'}

def trial(output):
    profile=json.loads((HERE/'profile.json').read_bytes());runtime=runtime_for(IM,(HERE/'package-lock.json').read_bytes(),profile,SYSTEM,TOOLS)
    build=lambda src:Snapshot.make({SHARED:SHARED_BYTES}|({COUNT:src.encode()} if src else {}),LOAD,runtime)
    h0,g,b=map(build,(None,GOOD,BAD));marker=secrets.token_hex(15)
    r=('read',dict(path='facts.txt',offset=1,limit=1));pos=('count_hits',dict(items=['hit','miss','hit']));zero=('count_hits',dict(items=['miss']))
    def op(name,**extra):return 'test_ops',dict(op=name,**extra)
    def rpc(name,**args):return 'ee_rpc',dict(operation=name,arguments=args)
    def write(src):return 'write',dict(path='../harness/'+COUNT,content=src)
    commit=rpc('harness_submit',paths=[SHARED,COUNT],load=LOAD)
    def activate(new,old):return rpc('harness_activate',version=new.identity,expected=old.identity)
    plans={
        'train2':[r,write(GOOD),commit,activate(g,h0),write(BAD),commit,activate(b,g)],
        'train0':[r,write(BAD),commit,op('install'),pos],
        'C1':[r,pos,zero,op('frozen_write'),write(GOOD),commit,activate(g,b),op('install'),zero,op('make_private',marker=marker),op('protected'),rpc('develop_quality',paths=[])],
        'stay':[r,zero], 'revert':[r,zero,activate(g,b)], 'reopened':[r,zero],
        'C2':[r,op('check_private'),pos,zero], 'B':[r,op('check_private'),zero,write(GOOD),op('install'),zero],
    }
    with tempfile.TemporaryDirectory(prefix='ee-v3-shared-pair-') as t:
        base=Path(t);lineage=Lineage(base/'lineage',runtime);lineage.initialize(h0); frozen=None
        contract=SourceContract(SOURCE,BASE,max_files=1,max_bytes=4096,max_text_growth=40,max_new_literal=45)
        for role,actions in plans.items():
            snap=h0 if role in ('train2','train0','B') else (lineage.active() if role=='reopened' else frozen)
            state={};trace=[];provider_requests=[];challenges={}
            def worker(op,c,lease):return DevReceipt(c.digest,op,'valid',15,'trusted-synthetic',{'synthetic':True})
            policy=Session(MODES[role],lineage,contract,FACT,worker,wall_ms=90000,node_ms=100,dev_calls=1,reservations={'develop_quality':80})
            def handler(name,args):
                if name=='toy_load':
                    h=args['digest'];source=args['origin'];expected=dict(snap.items).get(COUNT)
                    verify((source=='frozen' and expected and h==digest(expected) and not state) or (source=='local' and h in (digest(BAD.encode()),digest(GOOD.encode()))),'invalid loaded artifact')
                    state.update(digest=h,origin=source);trace.append(dict(kind='loaded',**state));return 200,dict(ok=True)
                if name=='toy_use':
                    verify({k:args[k] for k in state}==state and args['id']==f'call_p2_{len(provider_requests)}','unbound invocation')
                    n=args['args']['items'].count('hit');expected={'error':'TOY_ZERO_BUG'} if n==0 and args['digest']==digest(BAD.encode()) else dict(n=n)
                    verify(args['result']==expected,'unexpected actual tool output');s=secrets.token_hex(15);challenges[args['id']]=s
                    trace.append(dict(kind='invoked',**state,result=expected,rpc_challenge_sha=digest(s.encode())));return 200,dict(challenge=s)
                if name=='toy_obs':
                    opn=args['op'];value=args['result'];keys=['script','cache','log','session','notes'];
                    if opn=='frozen_write':verify(value=='EROFS','frozen artifact not read-only')
                    elif opn=='protected':verify(value==dict.fromkeys(['judge','sealed','ledger','receipt','protected_tool'],403),'protected routes visible')
                    elif opn=='make_private':verify(value=={k:digest((marker+':'+k).encode()) for k in keys},'private marks not created')
                    elif opn=='check_private':verify(value==dict.fromkeys(keys),'private state leaked')
                    else:raise ValueError('bad observation')
                    trace.append(dict(kind='observation',op=opn,value=value));return 200,dict(ok=True)
                return policy(name,args)
            @contextmanager
            def transport(raw,left):
                req=json.loads(raw);provider_requests.append(req);i=len(provider_requests)
                tools={t['function']['name'] for t in req['tools']};verify(('count_hits' in tools)==bool(state),'effective tool list incompatible with actual load')
                if i>1:
                    prior=actions[i-2];msg=[x for x in req['messages'] if x['role']=='tool'][-1];res=json.dumps(msg['content'])
                    if prior[0]=='count_hits':
                        c=challenges.get(msg['tool_call_id']);
                        if c:verify(c in res,'invocation evidence did not reach model')
                        else:verify(role=='B' and i==4 and ('not found' in res.lower() or 'unknown' in res.lower()),'unavailable tool not observed')
                    if prior[0]=='read':verify('toy prior fact' in res,'retrieval differs')
                if i==1:verify(marker not in json.dumps(req['messages']),'private fork state in initial request')
                yield chunks(i,actions[i-1] if i<=len(actions) else None).splitlines(keepends=True)
            model=MaxLease(Limits(230000,4096,25,90000,2048),transport)
            rep=sandbox(IM,SOURCE,snap,runtime,profile,SYSTEM,PROMPT,Gateway(model,handler),seconds=65,tools=TOOLS)
            first=provider_requests[0];common=digest(canonical(dict(tools=[x for x in first['tools'] if x['function']['name']!='count_hits'],messages=first['messages'])))
            output['roles'][role]=dict(sandbox=rep,trace=trace,policy=policy.rows,common_first=common,after=lineage.active().identity)
            verify(rep['status']=='pi_session_complete' and len(rep['pi_tool_executions'])==len(actions),'common runner did not finish '+role)
            if role=='train2':frozen=lineage.freeze();verify(frozen==b,'not the explicit active version')
            if role=='stay':verify(lineage.active()==b,'implicit rollback')
            if role=='revert':verify(lineage.active()==g,'explicit rollback missing')
            if role in ('C1','C2','B'):
                observed=[e['result'] for e in trace if e['kind']=='invoked'];want={'C1':[{'n':2},{'error':'TOY_ZERO_BUG'},{'n':0}], 'C2':[{'n':2},{'error':'TOY_ZERO_BUG'}],'B':[{'n':0}]}
                verify(observed==want[role], 'observed behavior differs')
        verify(len({output['roles'][x]['common_first'] for x in ('C1','C2','B')})==1,'paired initial Pi wire drift')
        verify(output['roles']['reopened']['trace'][-1]['result']=={'n':0},'rollback not effective next session')
        output['judge']=judge(base,secrets.token_hex(5));verify(output['judge']['result']['status']=='pass','clean Judge failed')
        output['frozen_version']=frozen.identity; output['active_training_version']=lineage.active().identity
    output['status']='pass_shared_runner_fixture'

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('out',type=Path);p=a.parse_args();r=dict(status='not_passed',scope='T1/T2/T3/T4/T5/T6/T7 scripted fixture on common Runner',roles={},paid_calls=0,gpu_calls=0)
    with p.out.open('x',encoding='utf8') as f:
        try:trial(r)
        except (ValueError,RuntimeError,OSError,KeyError,TypeError,IndexError) as e:r['failure']=type(e).__name__+': '+str(e)
        f.write(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(r['status'],r.get('failure',''));raise SystemExit(r['status']=='not_passed')
