"""P2 transport and lifecycle fixture: real Pi subprocesses; scripted localhost provider."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import re
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_cpu_wire.check import PI, PROFILE, MODEL, OWNER, FACT, sha, enc, need, clean_env

SOURCE=Path(__file__).with_name('h_bad.mjs')
TOOL='count_hits'


def chunk(turn, action):
    if action:
        name, args=action
        delta={'tool_calls':[{'index':0,'type':'function','id':f'call_toy_{turn}','function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]};finish='tool_calls'
    else:
        delta={'content':'FIXTURE_DONE'};finish='stop'
    parts=[]
    for d,f in [({'role':'assistant'},None),({'reasoning_content':'toy'},None),(delta,None),({},finish)]:
        part=dict(id='chatcmpl-fixture',object='chat.completion.chunk',created=1,model=MODEL,choices=[dict(index=0,delta=d,finish_reason=f)])
        parts.append(b'data: '+enc(part)+b'\n\n')
    usage=dict(id='chatcmpl-fixture',object='chat.completion.chunk',created=1,model=MODEL,choices=[],usage=dict(prompt_tokens=60,completion_tokens=12,total_tokens=72,completion_tokens_details={'reasoning_tokens':5}))
    return b''.join(parts)+b'data: '+enc(usage)+b'\n\ndata: [DONE]\n\n'


def session(base,label,trace,store=None, *, persistent_http=False):
    training=label.startswith('s1');has_h=store is not None
    home=base/label;ag=home/'pi';toy=home/'toy';ag.mkdir(parents=True);toy.mkdir()
    (toy/'facts.txt').write_bytes(FACT.encode()); source=SOURCE.read_bytes(); loaded_file=home/'frozen.mjs'
    if has_h:
        snapshot=store.read_bytes()
        need(sha(snapshot)==sha(source),'frozen snapshot differs from fixture')
        loaded_file.write_bytes(snapshot)  # copy stored S1 bytes, not a newly emitted fixture
        need(sha(loaded_file.read_bytes())==store.stem,'installed artifact hash differs from snapshot name')
    write_args=dict(path='bad.mjs',content=source.decode('utf8'))
    actions=[('write',write_args)] if training else [('read',dict(path='facts.txt',offset=1,limit=1))]
    if has_h:actions += [(TOOL,{'items':['hit','miss','hit']}),(TOOL,{'items':['miss']})]
    rows=[];events=[];errors=[];challenges={};trace.update(wire=rows,witness=events,boundary_errors=errors);lease=secrets.token_hex(24);model_key=secrets.token_hex(24);system='Use the toy tools only.'
    connections={}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1' if persistent_http else 'HTTP/1.0'
        def log_message(self,*unused):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'));need(0<size<110000,'request size')
                body=json.loads(self.rfile.read(size));auth=self.headers.get('Authorization')
                if self.path in ('/fixture/loaded','/fixture/invoked'):
                    need(has_h and auth=='Bearer '+lease,'unexpected gateway call')
                    if self.path.endswith('loaded'):
                        need(not events and body=={'fixture':'count-hits-bad-v1'},'bad load witness');kind='loaded'
                    else:
                        kind='invoked';expected=2 if len(events)==1 else 0;turn=len(events)+1
                        need(len(events) in (1,2) and body=={'id':f'call_toy_{turn}','n':expected},'bad invocation identity/value')
                    token=secrets.token_hex(16);events.append(dict(event=kind,**body,challenge_sha256=sha(token.encode())))
                    if kind=='invoked':challenges[body['id']]=token
                    payload=enc({'challenge':token});typ='application/json'
                else:
                    need(self.path=='/v1/chat/completions' and auth=='Bearer '+model_key,'bad model route')
                    turn=len(rows)+1;need(turn<=len(actions)+1,'turn cap')
                    need(body.get('model')==MODEL and body.get('thinking')=={'type':'enabled'} and body.get('reasoning_effort')=='max' and body.get('stream') is True and body.get('max_tokens')==PROFILE['maxTokens'],'wire drift')
                    tools=sorted(t['function']['name'] for t in body['tools']);want=sorted(['read','write']+([TOOL] if has_h else []));need(tools==want,'unexpected effective tools: '+str(tools))
                    if has_h:need(bool(events) and events[0]['event']=='loaded','no load witness')
                    msgs=body['messages'];conn=connections.setdefault(self.client_address,len(connections)+1)
                    row=dict(turn=turn,connection_ordinal=conn,tools=tools,public_tool_schema_sha256=sha(enc([t for t in body['tools'] if t['function']['name']!=TOOL])),max_verified=True)
                    if turn>1:
                        prev=actions[turn-2][0];cid=f'call_toy_{turn-1}';last=msgs[-1];need(last.get('role')=='tool' and last.get('tool_call_id')==cid,'tool reply not matched');text=json.dumps(last.get('content'),ensure_ascii=False)
                        if prev=='read':need(FACT in text,'shared fact not read')
                        if prev=='write':need((toy/'bad.mjs').read_bytes()==source,'real write did not produce exact source')
                        if prev==TOOL:
                            token=challenges.get(cid);prefix='COUNT=2:' if turn==3 else 'TOY_ZERO_BUG:'
                            need(token and prefix+token in text,'no live execution receipt');row['challenge_matched']=True
                        row['previous_tool']=prev
                    rows.append(row);payload=chunk(turn,actions[turn-1] if turn<=len(actions) else None);typ='text/event-stream'
                self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            except Exception as err:
                errors.append(str(err));self.send_error(400,'toy protocol mismatch')
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as srv:
        threading.Thread(target=srv.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{srv.server_port}'
        (ag/'models.json').write_bytes(enc({'providers':{OWNER:dict(baseUrl=url+'/v1',api='openai-completions',apiKey=model_key,models=[PROFILE])}}));(ag/'settings.json').write_bytes(enc(dict(compaction={'enabled':False},retry={'enabled':False,'provider':{'maxRetries':0}})))
        args=['node',str(PI),'--mode','json','--provider',OWNER,'--model',MODEL,'--thinking','max','--no-session','--no-extensions','--no-mcp','--no-context-files','--no-skills','--no-prompt-templates','--tools','read,write,count_*','--system-prompt',system]
        if has_h:args+=['-e',str(loaded_file)]
        env=clean_env(ag);env.update(PI_OFFLINE='1',EE_FIXTURE_GATE=url,EE_FIXTURE_LEASE=lease)
        try:proc=subprocess.run(args+['--print','Check the synthetic fixture.'],cwd=toy,env=env,stdin=subprocess.DEVNULL,capture_output=True,timeout=55)
        except subprocess.TimeoutExpired as e:
            partial=[json.loads(l) for l in (e.stdout or b'').splitlines() if l.startswith(b'{') and l.endswith(b'}')]
            trace['events_before_timeout']=[x.get('type') for x in partial if x.get('type')!='message_update']
            raise ValueError(label+': Pi timeout after 55s') from e
        finally:srv.shutdown()
    cli_events=[json.loads(l) for l in proc.stdout.splitlines() if l.startswith(b'{')]
    trace['exit_code']=proc.returncode
    trace['event_types']=[e['type'] for e in cli_events if isinstance(e,dict) and e.get('type') not in ('message_update',)]
    trace['assistant_stop_reasons']=[m.get('stopReason') for e in cli_events if isinstance((m:=e.get('message')),dict) and e.get('type')=='message_end' and m.get('role')=='assistant']
    trace['assistant_errors']=[]
    for event in cli_events:
        msg=event.get('message',{})
        if event.get('type')=='message_end' and isinstance(msg,dict) and msg.get('stopReason')=='error':
            # Raw prompts and tool arguments are never diagnostics. Only a single redacted error line.
            text=str(msg.get('errorMessage','unknown')).splitlines()[0][:320] if msg.get('errorMessage') else 'unknown'
            for private in (lease,model_key,str(home),str(base)):
                text=text.replace(private,'[fixture]')
            text=re.sub(r'(?i)bearer\s+\S+|[0-9a-f]{32,}', '[redacted]',text)
            trace['assistant_errors'].append(text)
    need(not errors,label+': '+str(errors));need(proc.returncode==0 and len(rows)==len(actions)+1,label+f': incomplete wire, rc={proc.returncode}, turns={len(rows)}')
    ends=[(e.get('toolName'),e.get('isError')) for e in cli_events if e.get('type')=='tool_execution_end'];expected=[(a[0],has_h and i==2) for i,a in enumerate(actions)];need(ends==expected and b'FIXTURE_DONE' in proc.stdout,'tool event mismatch: '+str(ends))
    report=dict(session=label,private_home=True,pi_exit=proc.returncode,wire=rows,tool_ends=ends,witness=events,shared_fact_sha256=sha((toy/'facts.txt').read_bytes()),snapshot_sha256=sha(source) if has_h else None)
    written=(toy/'bad.mjs').read_bytes() if training else None
    return report,written


def run(traces, persistent_http=False):
    version=subprocess.check_output(['node',str(PI),'--version'],text=True,timeout=15).strip();need(version=='1.5.2','Pi drift');history=[]
    with tempfile.TemporaryDirectory(prefix='ee-tool-restore-') as t:
        base=Path(t);store=base/'snapshots';store.mkdir();artifact=None
        for label in ('s1-B2','s1-B0','C1','B','C2'):
            # Each session directory is lifecycle-owned, freshly created and disposed by TemporaryDirectory.
            with tempfile.TemporaryDirectory(dir=base,prefix='session-') as w:
                traces[label]={}
                report,artifact_bytes=session(Path(w),label,traces[label],artifact if label in ('C1','C2') else None,persistent_http=persistent_http)
                if label=='s1-B2':
                    artifact=store/(sha(artifact_bytes)+'.mjs');artifact.write_bytes(artifact_bytes)
                history.append(report)
            need(not Path(w).exists(),'session not torn down')
        need(artifact.read_bytes()==SOURCE.read_bytes() and len(list(store.iterdir()))==1,'frozen artifact changed')
        base_hashes={row['public_tool_schema_sha256'] for s in history for row in s['wire']};need(len(base_hashes)==1,'common tool schemas drifted')
    return dict(status='pass_scoped',pi_offline=True,persistent_http=persistent_http,scope='fixed scripted fixture, not autonomous authorship or security attestation',pi=version,fixture_sha256=sha(SOURCE.read_bytes()),paid_calls=0,gpu_calls=0,sessions=history)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--persistent-http',action='store_true');args=ap.parse_args();f=args.receipt;need(not f.exists(),'receipt exists')
    traces={}
    try:j=run(traces,args.persistent_http)
    except Exception as e:j=dict(status='not_passed',reason=str(e),partial_evidence=traces)
    with f.open('x',encoding='utf8') as out:json.dump(j,out,indent=2,ensure_ascii=False)
    print(j['status'],j.get('reason',''));sys.exit(j['status']!='pass_scoped')
