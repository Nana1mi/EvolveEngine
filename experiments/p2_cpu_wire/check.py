"""Partial P2/T3: real Pi, paired synthetic Harness prompts, local mock DeepSeek SSE only."""
import argparse, hashlib, json, os, secrets, subprocess, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]; PI=REPO/'.evolveengine/pi-smoke/node_modules/@earendil-works/pi-coding-agent/dist/cli.js'
MODEL='deepseek-flash'; FACT='toy-fact-score-31'; MARK='toy-H-instruction-6ba'; THINK=['mock-分段','\nsecond segment']; OWNER='wire'; SYSTEM='Read only the supplied toy fact.'
PROFILE=dict(id=MODEL,name='DeepSeek V4.1 Flash',reasoning=True,thinkingLevelMap=dict(minimal=None,low='low',medium=None,high='high',max='max'),contextWindow=1000000,maxTokens=4096,compat=dict(supportsStore=False,supportsDeveloperRole=False,maxTokensField='max_tokens',thinkingFormat='deepseek',requiresReasoningContentOnAssistantMessages=True,supportsReasoningEffort=True))

def need(x,reason):
    if not x:raise ValueError(reason)
def enc(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def clean_env(ag):
    keep={'PATH','PATHEXT','SYSTEMROOT','WINDIR','COMSPEC','TEMP','TMP','TMPDIR','LANG','LC_ALL'}
    return {**{k:v for k,v in os.environ.items() if k.upper() in keep},'USERPROFILE':str(ag.parent),'HOME':str(ag.parent),'PI_CODING_AGENT_DIR':str(ag),'PI_SKIP_VERSION_CHECK':'1','NO_PROXY':'127.0.0.1,localhost'}

def wire(req,has_h,turn):
    need(req.get('model')==MODEL and req.get('stream') is True and req.get('thinking')=={'type':'enabled'} and req.get('reasoning_effort')=='max' and req.get('max_tokens')==PROFILE['maxTokens'],'model/Max/output cap/stream mismatch')
    msgs=req['messages']; tools=req['tools'];need([t['function']['name'] for t in tools]==['read'],'common tool mismatch')
    system=[m['content'] for m in msgs if m['role']=='system']; s=json.dumps(system,ensure_ascii=False);need(SYSTEM in s and (MARK in s)==has_h,'incorrect effective Harness prompt')
    row=dict(turn=turn,model=MODEL,thinking='enabled',reasoning_effort='max',max_tokens=req.get('max_tokens'),toolset_sha256=sha(enc(tools)),h_marker_present=MARK in s, common_system_sha256=sha(enc(system).replace((' '+MARK).encode(),b'')),user_messages_sha256=sha(enc([m['content'] for m in msgs if m['role']=='user'])))
    if turn==2:
        assistants=[m for m in msgs if m['role']=='assistant' and m.get('tool_calls')]; need(len(assistants)==1,'missing prior tool call'); a=assistants[0]; value=a.get('reasoning_content'); need(isinstance(value,str) and value.encode()==''.join(THINK).encode(),'reasoning_content changed')
        calls=a['tool_calls'];responses=[m for m in msgs if m['role']=='tool']; need(len(calls)==len(responses)==1 and calls[0]['id']==responses[0]['tool_call_id'] and FACT in json.dumps(responses[0]['content']),'read not completed')
        row.update(synthetic_reasoning_exact=True, synthetic_reasoning_bytes=len(value.encode()), fact_returned=True)
    return row

def packets(turn):
    delta=[{'role':'assistant'}]; finish='stop'
    if turn==1:
        delta.extend({'reasoning_content':v} for v in THINK); delta.append({'tool_calls':[{'index':0,'type':'function','id':'call_toy_1','function':{'name':'read','arguments':json.dumps(dict(path='facts.txt',offset=1,limit=1))}}]});finish='tool_calls'
    else:delta.append({'content':'WIRE_OK'})
    for d in delta: yield dict(choices=[dict(index=0,delta=d,finish_reason=None)])
    yield dict(choices=[dict(index=0,delta={},finish_reason=finish)])
    yield dict(choices=[],usage=dict(prompt_tokens=60,completion_tokens=12,total_tokens=72,completion_tokens_details={'reasoning_tokens':5}))

def paired():
    version=subprocess.check_output(['node',str(PI),'--version'],timeout=15,text=True).strip();need(version=='1.5.2','Pi changed'); result={}
    with tempfile.TemporaryDirectory(prefix='ee-p2-pair-') as path:
        tmp=Path(path);root=tmp/'toy'; root.mkdir(); (root/'facts.txt').write_text(FACT,encoding='utf8');pw=secrets.token_hex(24)
        for role in ('B0','B2-fixture'):
            ag=tmp/role/'pi';ag.mkdir(parents=True); rows=[];errors=[];has_h=role!='B0'
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_POST(self):
                    try:
                        need(self.path=='/v1/chat/completions' and self.headers.get('Authorization')=='Bearer '+pw,'mock endpoint mismatch');size=int(self.headers.get('Content-Length','0'));need(0<size<65536 and len(rows)<2,'request limit');row=wire(json.loads(self.rfile.read(size)),has_h,len(rows)+1);rows.append(row)
                        parts=[b'data: '+enc(dict(id='chatcmpl-test',object='chat.completion.chunk',created=1,model=MODEL,**x))+b'\n\n' for x in packets(row['turn'])]+[b'data: [DONE]\n\n'];out=b''.join(parts)
                        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(out)));self.end_headers();self.wfile.write(out)
                    except Exception as err:
                        errors.append(str(err));self.send_error(400,'fixture rejected request')
            with ThreadingHTTPServer(('127.0.0.1',0),Handler) as srv:
                threading.Thread(target=srv.serve_forever,daemon=True).start()
                (ag/'models.json').write_bytes(enc({'providers':{OWNER:dict(baseUrl=f'http://127.0.0.1:{srv.server_port}/v1',api='openai-completions',apiKey=pw,models=[PROFILE])}})); (ag/'settings.json').write_bytes(enc(dict(compaction={'enabled':False},retry={'enabled':False,'provider':{'maxRetries':0}})))
                args=['node',str(PI),'--mode','json','--provider',OWNER,'--model',MODEL,'--thinking','max','--no-session','--no-extensions','--no-mcp','--no-skills','--no-context-files','--no-prompt-templates','--tools','read','--system-prompt',SYSTEM+(' '+MARK if has_h else ''),'--print','Read facts.txt then finish.']
                try:proc=subprocess.run(args,cwd=root,env=clean_env(ag),stdin=subprocess.DEVNULL,capture_output=True,timeout=45)
                finally:srv.shutdown()
            need(not errors,str(errors));need(proc.returncode==0 and len(rows)==2,f'Pi rc={proc.returncode}, mock requests={len(rows)}');events=[json.loads(l) for l in proc.stdout.splitlines() if l.startswith(b'{')]; tool=[x for x in events if x.get('type')=='tool_execution_end'];need(len(tool)==1 and tool[0].get('toolName')=='read' and tool[0].get('isError') is False and b'WIRE_OK' in proc.stdout,'expected successful tool and completion');result[role]=rows
    shared=lambda r:[{k:v for k,v in x.items() if k!='h_marker_present'} for x in r];need(shared(result['B0'])==shared(result['B2-fixture']),'paired wire settings differ')
    return dict(status='pass_scoped',scope='P2/T3 transport fixture only; static Harness marker, no training, no sandbox claim',pi=version,pi_entry_sha256=sha(PI.read_bytes()),mock_upstream=True,paid_calls=0,arms=result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path);f=parser.parse_args().receipt
    try:rep=paired()
    except Exception as e:rep=dict(status='not_passed',reason=str(e),mock_upstream=True)
    with f.open('x',encoding='utf8') as s:json.dump(rep,s,indent=2,ensure_ascii=False)
    print(rep['status'],rep.get('reason',''));raise SystemExit(rep['status']!='pass_scoped')
