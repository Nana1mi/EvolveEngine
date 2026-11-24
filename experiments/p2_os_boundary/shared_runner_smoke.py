"""Exercise the shared V3 runner through Docker with real Pi and a fake upstream."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from broker import chunks
from evolveengine.v3_harness import Lineage, Snapshot
from evolveengine.v3_provider import MaxLease, Limits
from evolveengine.v3_runner import Gateway, Images, runtime_for, sandbox
from evolveengine.v3_session import DEVELOP, DevReceipt, Session, SourceContract

SYSTEM='Optimize the supplied synthetic Engine subject to development validation; submit only measured, accepted exact source.'
PROMPT='Use the development quality tool, then submit the exact source you wish to deliver.'
EDIT={'files':{'engine.py':'def count(xs):\n    return xs.count("hit")\n'}}
BASE={'engine.py':b'def count(xs):\n    return sum(x=="hit" for x in xs)\n'}
FACT={'prior':'toy fact; not sealed'}
EMPTY=dict(prompts=[],skills=[],extensions=[],workflows=[],memory=[])
MARK='This frozen harness carries an extra toy note, not training facts.'
TOOLS=['read','write','edit','bash','ee_rpc']
IM=Images('sha256:efe1685cb8525c933297166615061df41d6735bc927cd6650c31354bb0afc41e','sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df')


def verify(ok,msg):
    if not ok:raise ValueError(msg)

def run(out):
    profile=json.loads((HERE/'profile.json').read_bytes()); runtime=runtime_for(IM,(HERE/'package-lock.json').read_bytes(),profile,SYSTEM,TOOLS)
    contract=SourceContract(BASE,BASE,max_files=1,max_bytes=1024,max_text_growth=32,max_new_literal=25); candidate=contract.from_args(EDIT)
    with tempfile.TemporaryDirectory(prefix='ee-v3-runner-smoke-') as t:
        lineage=Lineage(t,runtime);h0=Snapshot.make({},EMPTY,runtime);lineage.initialize(h0)
        h2=lineage.get(lineage.submit('evolving_train',{'prompts/note.md':MARK.encode()},dict(EMPTY,prompts=['prompts/note.md'])))
        for role,snap in [('fixed',h0),('frozen',h2)]:
            called=[]; observed=[]
            @contextmanager
            def transport(raw,left):
                req=json.loads(raw);called.append(req);turn=len(called)
                actions=[('read',dict(path='engine.py',offset=1,limit=2)),('ee_rpc',dict(operation='develop_quality',arguments=EDIT)),
                         ('ee_rpc',dict(operation='deliver_engine',arguments=EDIT))]
                if turn>1:
                    tool=[m for m in req['messages'] if m['role']=='tool'][-1]; body=json.dumps(tool['content'])
                    verify(('sum' if turn==2 else ('valid' if turn==3 else 'signed')) in body,'scripted provider did not receive real tool result')
                yield chunks(turn, actions[turn-1] if turn<=len(actions) else None).splitlines(keepends=True)
            def worker(op,engine,max_ms):
                verify(op=='develop_quality' and engine==candidate,'worker received wrong artifact');observed.append(engine.digest)
                return DevReceipt(engine.digest,op,'valid',17,'controlled-external-quality',{'synthetic':True})
            outer=Session('evaluation',lineage,contract,FACT,worker,wall_ms=90000,node_ms=1000,dev_calls=1,reservations=dict.fromkeys(DEVELOP,200))
            provider=MaxLease(Limits(input=40000,output=3000,turns=6,wall_ms=90000,per_turn_output=2000),transport,
                              expected_prompt_fragments=() if role=='fixed' else (MARK,))
            s=sandbox(IM,BASE,snap,runtime,profile,SYSTEM,PROMPT,Gateway(provider,outer),seconds=55,tools=TOOLS)
            out['arms'][role]=dict(s,external_rows=outer.rows,input_tokens=provider.spent_input,output_tokens=provider.spent_output)
            verify(s['status']=='pi_session_complete',s['status']+'; provider rows='+str(len(s['provider_rows'])))
            verify(outer.delivered==candidate and observed==[candidate.digest] and [x['name'] for x in s['pi_tool_executions']]==['read','ee_rpc','ee_rpc'],'external policy round-trip missing')
            verify(len(called)==4 and provider.spent_output==48 and provider.spent_input==240,'model protocol costs inconsistent')
        a,b=(out['arms'][x]['provider_rows'][0] for x in ('fixed','frozen'))
        verify(a['toolset_sha']==b['toolset_sha'] and a['prompt_fragments']==[] and len(b['prompt_fragments'])==1,'shared runner paired evidence drift')
    out['status']='pass_scoped'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();r=dict(status='not_passed',scope='shared V3 runner with isolated Pi; no real GPU/network provider',arms={},paid_calls=0,gpu_calls=0)
    with a.out.open('x',encoding='utf8') as f:
        try:run(r)
        except (ValueError,OSError,RuntimeError,KeyError,TypeError) as e:r['reason']=str(e)
        f.write(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(r['status'], r.get('reason',''));raise SystemExit(r['status']!='pass_scoped')
