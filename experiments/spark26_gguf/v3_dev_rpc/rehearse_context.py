"""Offline-only P3 context rehearsal: real pinned Pi, synthetic SSE, no live transport."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile

from p3 import (CAP, EMPTY, PI, SMOKE, SYSTEM, TRAIN_MODES, Gateway, Lineage, Snapshot, Session,
                SourceContract, canonical, digest, need, pilot_profile, runtime_for, sandbox)
from evolveengine.v3_context_control import ARTIFACTS, MODEL_REVISION, TOKENIZERS_VERSION, VERSION, OfflineContextLease, PublicPromptCounter

ROOT=Path(__file__).resolve().parents[3]
FILE='toy.py'
TEXT="\ufeff# "+('x'*37+' quote=" \\ \u96ea \U0001f680\r\n')*430+'last\n'
BYTES=TEXT.encode();FP='context-offline-fixture'


def response(i,action=None):
    def evt(delta=None,finish=None,usage=None):
        return b'data: '+canonical(dict(model='deepseek-flash',system_fingerprint=FP,
                        choices=[dict(index=0,delta=delta or {},finish_reason=finish)],usage=usage))+b'\n\n'
    a=evt(dict(role='assistant',reasoning_content='offline '))+evt(dict(reasoning_content=f'thought {i}\n\u96ea'))
    if action is not None:a+=evt(dict(tool_calls=[dict(index=0,id=f'context_fixture_{i}',type='function',function=dict(name='read',arguments=canonical(action).decode()))]))
    else:a+=evt(dict(content='fixture finished'))
    return a+evt(finish='tool_calls' if action is not None else 'stop',usage=dict(prompt_tokens=100,completion_tokens=30,total_tokens=130,completion_tokens_details=dict(reasoning_tokens=20)))+b'data: [DONE]\n\n'


def pages(root):
    file=root/FILE;file.write_bytes(BYTES);module=(ROOT/'evolveengine/v3_paged_read.mjs').as_uri()
    js='''import {page} from MODULE;
import {appendFile} from 'node:fs/promises';
const p=process.argv[1], output=[];let at=0,version;
for(;;){const part=await page(p,at,version);output.push(part);if(!part.truncated)break;at=part.next_offset;version=part.version;}
await appendFile(p,'changed');
let changed=false; try{await page(p,0,output[0].version);}catch(e){changed=e.message.startsWith('File changed');}
let split=false; try{await page(p,1);}catch(e){split=true;}
process.stdout.write(JSON.stringify({output,changed,split}));'''.replace('MODULE',json.dumps(module))
    p=subprocess.run(['node','--input-type=module','-e',js,str(file)],capture_output=True,check=True,timeout=15)
    checks=json.loads(p.stdout);parts=checks['output']
    need(checks['changed'] and checks['split'] and len(parts)>1 and ''.join(x['content'] for x in parts).encode()==BYTES,
         'direct paging was lossy or accepted a stale or invalid offset')
    return parts


def run(out,assets):
    counter=PublicPromptCounter(assets);profile=pilot_profile();tools=['read'];prompt='Exercise the shared text reader.'
    runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),profile,SYSTEM,tools,paged_read=True)
    out.update(protocol=VERSION,model_revision=MODEL_REVISION,artifacts=dict(ARTIFACTS),tokenizers=TOKENIZERS_VERSION,
               limits=asdict(CAP),runtime=runtime,cases={},pinned_sources={n:digest((ROOT/'evolveengine'/n).read_bytes()) for n in
                ('v3_provider.py','v3_context_control.py','v3_paged_read.mjs','v3_runner.py','v3_guest.mjs')})
    with tempfile.TemporaryDirectory(prefix='ee-p3-context-check-') as tmp:
        root=Path(tmp);segments=pages(root)
        responses=[response(i,dict(path=FILE,offset=x['offset'])) for i,x in enumerate(segments,1)]+[response(len(segments)+1)]
        def worker(*_):raise AssertionError('offline fixture cannot request GPU')
        for arm,mode in TRAIN_MODES.items():
            print(arm,'paged replay',flush=True)
            source={FILE:BYTES};h=Lineage(root/arm,runtime);snap=Snapshot.make({},EMPTY,runtime);h.initialize(snap)
            contract=SourceContract(source,source,max_files=1,max_bytes=160*1024,max_text_growth=1024,max_new_literal=1024)
            policy=Session(mode,h,contract,{},worker,wall_ms=CAP.wall_ms,node_ms=800000,dev_calls=4,reservations={'develop_quality':200000,'develop_measure':200000})
            observed=[]
            class Inspect:
                def reservation(self,req):
                    index=len(observed)
                    tool,=[t for t in req['tools'] if t['function']['name']=='read']
                    need('expected_version' in tool['function']['parameters']['properties'],'built-in read was not replaced')
                    if index:
                        last=req['messages'][-1];need(last['role']=='tool','Pi did not replay the tool response')
                        content=last['content'];length=len(content.encode());part=json.loads(content)
                        need(length<=8192 and all(part[k]==segments[index-1][k] for k in ('offset','next_offset','truncated','content')),
                             'Pi page was lossy, oversized, or incorrectly continued')
                        need(len(part['version'])==64,'read resume marker missing')
                        if index>1:need(part['version']==observed[-1]['version'],'file identity changed across actual Pi pages')
                        observed.append(dict(part,bytes=length))
                    else:observed.append(None)
                    return counter.reservation(req)
            lease=OfflineContextLease(CAP,Inspect(),responses=responses,fingerprint=FP)
            runner=sandbox(PI,source,snap,runtime,profile,SYSTEM,prompt,Gateway(lease,policy),seconds=75,tools=tools,paged_read=True)
            need(runner['status']=='pi_session_complete' and runner['isolated_state_removed'] and not runner['isolation']['gpu']
                 and not lease.rejections and len(lease.rows)==len(responses) and not policy.rows and
                 runner['pi_tool_executions']==[dict(name='read',error=False)]*len(segments),'actual Pi paging or reasoning replay did not pass')
            need(''.join(x['content'] for x in observed[1:]).encode()==BYTES,'actual Pi pages did not concatenate to source')
            out['cases'][arm]=dict(mode=mode,finish=[r['finish'] for r in lease.rows],tool_pages=len(segments),
                 payload_bytes=[x['bytes'] for x in observed[1:]],local_counts=[r['request_meta']['input_reservation'] for r in lease.rows],
                 replay=[r['request_meta']['replay'] for r in lease.rows],source_sha=digest(BYTES),source_bytes=len(BYTES),lossless=True)
    out.update(status='pass_offline_candidate_only',hosted_input_upper_bound='not_attested',paid_calls=0,gpu_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);p.add_argument('receipt',type=Path);args=p.parse_args()
    out=dict(status='not_passed')
    with args.receipt.open('x',encoding='utf-8',newline='\n') as f:
        try:run(out,args.assets)
        except (ValueError,RuntimeError,OSError,AssertionError,TypeError,KeyError) as e:out['error']=str(e)
        finally:f.write(json.dumps(out,indent=2)+'\n')
    print(out['status'],out.get('error',''));raise SystemExit(out['status']!='pass_offline_candidate_only')
