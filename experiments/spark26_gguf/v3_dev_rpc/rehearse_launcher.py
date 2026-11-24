"""Rehearse the actual launcher's initial tool discovery without paid calls/GPU."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile

from p3 import PI,EMPTY,SMOKE,SYSTEM,PROMPT, runtime_for,sandbox,Session,SourceContract,Gateway,MaxLease,Limits,Snapshot,Lineage,need,execution_protocol,pilot_profile,provider_diagnostics
from verify_common import chunks


def run(result):
    profile=pilot_profile();runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),profile,SYSTEM,None)
    engine={'gguf.py':b'def toy():\n    return 1\n'};rows=[]
    with tempfile.TemporaryDirectory(prefix='ee-v3-launch-') as root:
        h=Lineage(root,runtime);snap=Snapshot.make({},EMPTY,runtime);h.initialize(snap)
        @contextmanager
        def provider(raw,_left):
            j=json.loads(raw);rows.append(j);n=len(rows)
            tools={x['function']['name'] for x in j['tools']}
            need({'read','write','bash','edit','ee_rpc'}<=tools,'default open tooling was not available')
            if n==2:
                r=[x['content'] for x in j['messages'] if x['role']=='tool'];need(r and snap.identity in json.dumps(r[-1]),'active Harness was not discoverable by Agent')
            action=('ee_rpc',dict(operation='harness_status',arguments={})) if n==1 else None
            yield chunks(n,action).splitlines(keepends=True)
        policy=Session('evolving_train',h,SourceContract(engine,engine,max_files=1,max_bytes=1000,max_text_growth=20,max_new_literal=20),{},
                       lambda *_:None,wall_ms=60000,node_ms=200000,dev_calls=1,reservations={'develop_quality':200000})
        model=MaxLease(Limits(20000,2000,4,60000,1000),provider)
        result.update(session=sandbox(PI,engine,snap,runtime,profile,SYSTEM,PROMPT,Gateway(model,policy),seconds=45,tools=None),trusted_rows=policy.rows,runtime=runtime,execution_protocol=execution_protocol(),provider_diagnostics=provider_diagnostics())
        need(result['session']['status']=='pi_session_complete' and [e['operation'] for e in policy.rows]==['harness_status'],'live-mode tool availability rehearsal did not finish')
        result['status']='pass_launcher_mock_scope'

if __name__=='__main__':
    p=Path(sys.argv[1]);j=dict(status='not_passed',scope='pilot launcher mode with mock upstream; not a free pilot',gpu_calls=0,paid_calls=0)
    with p.open('x',encoding='utf8') as f:
        try:run(j)
        except (ValueError,OSError,RuntimeError) as e:j['error']=str(e)
        f.write(json.dumps(j,indent=2)+'\n')
    print(j['status'],j.get('error',''));raise SystemExit(j['status']=='not_passed')
