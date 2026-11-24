"""One bounded, synthetic real Flash Max session on the common isolated V3 Runner.

Uses DEEPSEEK_API_KEY only in the trusted host process, never the Agent or relay. No GPU.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from evolveengine.deepseek import Provider
from evolveengine.v3_harness import Snapshot
from evolveengine.v3_provider import MaxLease, Limits
from evolveengine.v3_runner import Gateway, sandbox, runtime_for
from shared_runner_smoke import IM, EMPTY, verify

SYS='This is a synthetic tool protocol check. Do not change files or perform other work.'
TASK='First use read with exactly {"path":"facts.txt","offset":1,"limit":2}. Next use ee_rpc with operation "facts" and arguments {}. After both tool results, reply with just the two numbers in the order received. Do not combine these tool calls.'
CAP=Limits(input=22000,output=2600,turns=4,wall_ms=90000,per_turn_output=1000)
TOOLS=['read','ee_rpc']


def main(out):
    profile=json.loads((HERE/'profile.json').read_bytes());runtime=runtime_for(IM,(HERE/'package-lock.json').read_bytes(),profile,SYS,TOOLS)
    provider=Provider('https://api.deepseek.com','deepseek-flash','DEEPSEEK_API_KEY','DeepSeek-V4.1-Flash',CAP.per_turn_output)
    out['verified_model_listing']=provider.preflight()
    def transport(raw,remaining):return provider.request('/chat/completions',payload=raw,timeout=remaining)
    lease=MaxLease(CAP,transport);ops=[]
    def rpc(name,args):
        if name=='facts' and args=={}:
            ops.append('facts');return 200,dict(fact='The external toy number is 86.')
        return 403,dict(status='denied')
    try:
        s=sandbox(IM,{'facts.txt':b'The local toy number is 47.\n'},Snapshot.make({},EMPTY,runtime),runtime,profile,SYS,TASK,Gateway(lease,rpc),seconds=65,tools=TOOLS)
        out['session']=s
        verify(s['status']=='pi_session_complete' and s['pi_tool_executions']==[dict(name='read',error=False),dict(name='ee_rpc',error=False)] and ops==['facts'],'script did not complete both real tool calls')
        verify(len(lease.rows)==3 and [x['prior_tool_contexts'] for x in lease.rows]==[0,1,2]
              and all(r['usage']=='settled' and r['effort']=='max' for r in lease.rows),'provider protocol or meter evidence incomplete')
        verify(len({tuple(r['versions']) for r in lease.rows})==1,'model provider fingerprint changed')
        out['status']='pass_scoped'
    finally:
        out['provider_rows']=lease.rows;out['spent_input']=lease.spent_input;out['spent_output']=lease.spent_output;out['trusted_rpc']=ops

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('out',type=Path);args=a.parse_args();r=dict(status='not_passed',gpu_calls=0,scope='one bounded synthetic shared runner session, not a free optimization pilot')
    with args.out.open('x',encoding='utf8') as f:
        try:main(r)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError) as e:r['failure']=type(e).__name__+': '+str(e)[:180]
        f.write(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(r['status'],r.get('failure',''));raise SystemExit(r['status']!='pass_scoped')
