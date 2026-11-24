"""Connect the same real Pi Runner and tool broker to actual DEV quality and measurement."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2];sys.path.insert(0,str(REPO))
from evolveengine.v3_harness import Lineage,Snapshot
from evolveengine.v3_provider import MaxLease,Limits
from evolveengine.v3_runner import Gateway,Images,sandbox,runtime_for
from evolveengine.v3_session import DevReceipt,Session,SourceContract
from audit_worker import audit
from node_client import ALLOWED,connection,invoke,need
from reference_qkv import edited
sys.path.insert(0,str(REPO/'experiments/p2_os_boundary'))
from broker import chunks

PI=Images('sha256:efe1685cb8525c933297166615061df41d6735bc927cd6650c31354bb0afc41e','sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df')
BASE_SYSTEM='Exercise the isolated development workflow on authorized synthetic Spark test loads. Only a signed candidate can be delivered.'
TASK='Run the fixed four-operation synthetic conformance check; this is not an optimization task.'
TOOLS=['read','write','ee_rpc']
EMPTY=dict(prompts=[],skills=[],extensions=[],workflows=[],memory=[])
RESERVE=200000


def main(c,source,archive,out):
    source_bytes=source.read_bytes();path=ALLOWED[0];base={path:source_bytes};patch=edited(source_bytes).decode();paths=dict(paths=[path]);arg={'files':{path:patch}}
    actions=[('write',dict(path=path,content=patch))]
    actions.extend(('ee_rpc',dict(operation=op,arguments=paths)) for op in ('develop_quality','develop_measure','deliver_engine'))
    profile=json.loads((REPO/'experiments/p2_os_boundary/profile.json').read_bytes()); runtime=runtime_for(PI,(REPO/'experiments/p2_os_boundary/package-lock.json').read_bytes(),profile,BASE_SYSTEM,TOOLS)
    model_turns=[];receipts=[]
    @contextmanager
    def transport(raw,left):
        req=json.loads(raw);model_turns.append(req);turn=len(model_turns)
        if turn>=3:
            observed=[m for m in req['messages'] if m['role']=='tool'][-1]['content'];expected=('valid','observed','signed')[turn-3]
            need(expected in json.dumps(observed) and 'insufficient_evidence' not in json.dumps(observed),'real development receipt not returned to Pi')
        yield chunks(turn,actions[turn-1] if turn<=len(actions) else None).splitlines(keepends=True)
    def worker(op,artifact,max_ms):
        receipt={'operation':op,'source_identity':artifact.digest};receipts.append(receipt)
        root,rc=invoke(c,op,dict(artifact.edits),artifact.digest,base,max_ms)
        receipt.update(archive=str(root),rc=rc)
        need(rc==0,'worker process returned insufficient evidence')
        result=audit(root,archive/'spark26-v32-g1-d3.lv2uVT',archive/'g1-dev-fJPjW8');receipt.update(result=result)
        public={k:result[k] for k in ('scenes','max_tv','eos_passed','mixed_passed') if k in result}
        return DevReceipt(artifact.digest,op,result['status'],result['node_ms'],result['raw_sha'],public)
    with tempfile.TemporaryDirectory(prefix='ee-v3-actual-rpc-') as t:
        l=Lineage(t,runtime);snap=Snapshot.make({},EMPTY,runtime);l.initialize(snap)
        contract=SourceContract(base,[path],max_files=1,max_bytes=160*1024,max_text_growth=1024,max_new_literal=1024)
        outer=Session('evaluation',l,contract,{},worker,wall_ms=490000,node_ms=RESERVE*2,dev_calls=2,reservations={'develop_quality':RESERVE,'develop_measure':RESERVE})
        model=MaxLease(Limits(250000,4096,6,490000,1200),transport)
        try:
            s=sandbox(PI,base,snap,runtime,profile,BASE_SYSTEM,TASK,Gateway(model,outer),seconds=465,tools=TOOLS);out['session']=s
            need(s['status']=='pi_session_complete' and len(model.rows)==5 and len(receipts)==2 and outer.delivered==contract.from_args(arg),'Pi-to-real-worker chain did not complete')
            out['status']='pass_integrated_dev_scope'
        finally:
            out.update(worker=receipts,external_policy=outer.rows,spent_node_ms=outer.spent_node,model_rows=model.rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out=a.out.resolve();j=dict(status='not_passed',scope='scripted Pi tool calls to true Spark development Worker, no paid calls or free optimization')
    with a.out.open('x',encoding='utf8') as f:
        try:
            with connection() as c:main(c,a.source,Path(os.environ['LOCALAPPDATA'])/'EvolveEngine',j)
        except (OSError,ValueError,RuntimeError,TypeError,KeyError) as e:j['failure']=type(e).__name__+': '+str(e)
        f.write(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    print(j['status'],j.get('failure',''));raise SystemExit(j['status']=='not_passed')
