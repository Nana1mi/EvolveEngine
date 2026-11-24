"""Trusted external mock model, version gate, development service and witness."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import os
import secrets
import socketserver
import threading


def enc(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def sha(blob):
    if isinstance(blob,str): blob=blob.encode()
    return sha256(blob).hexdigest()


TOOLS = ('read','write','fixture_ops')
REASON = 'fixture analysis, no real reasoning or billing'
PROTECTED = {'judge','sealed','ledger','protected_tool','receipt'}


def need(value,msg):
    if not value: raise ValueError(msg)


def chunks(turn, action):
    result={'content':'P2_FIXTURE_DONE'}; finish='stop'
    if action:
        name,args=action
        result={'tool_calls':[dict(index=0,type='function',id=f'call_p2_{turn}',function=dict(name=name,arguments=enc(args).decode()))]}
        finish='tool_calls'
    def make(delta, done=None):
        return dict(id='completion-toy',object='chat.completion.chunk',created=1,model='deepseek-flash',choices=[dict(index=0,delta=delta,finish_reason=done)])
    parts=[make({'role':'assistant'}),make({'reasoning_content':REASON}),make(result),make({},finish)]
    parts.append(dict(id='completion-toy',object='chat.completion.chunk',created=1,model='deepseek-flash',choices=[],usage=dict(prompt_tokens=60,completion_tokens=12,total_tokens=72,completion_tokens_details={'reasoning_tokens':5})))
    return b''.join(b'data: '+enc(p)+b'\n\n' for p in parts)+b'data: [DONE]\n\n'


class Boundary:
    def __init__(self,spec,emit,protected_hashes,develop):
        self.protected_hashes=protected_hashes;self.develop=develop
        self.spec=spec; self.role=spec['role']; self.emit=emit;self.turn=0;self.n=0
        self.catalog=dict(spec['catalog']);self.pointer=spec['pointer'];self.frozen=spec['frozen']
        self.installed=None;self.expect=None;self.e1={};self.lock=threading.Lock();self.complete=False

    def event(self,kind,**fields):
        self.n+=1; self.emit(dict(run_id=self.spec['run_id'],phase=self.spec['phase'],session_id=self.role,fork_id=self.role,event_id=self.n,kind=kind,**fields))

    def handle(self,path,body,auth=None):
        with self.lock:
            return self.dispatch(path,body,auth)

    def dispatch(self,path,body,auth):
        if path=='/v1/chat/completions':
            return self.model(body,auth)
        op=path.removeprefix('/')
        if op in PROTECTED or (op in ('commit','activate') and self.spec['phase']!='train'):
            self.event('denied',operation=op,units=1)
            return 403,dict(denied=op)
        if op=='commit':
            slot=body['slot'];source=body['source']; digest=sha(source)
            need(slot in self.spec['variants'] and self.spec['variants'][slot]==digest,'unexpected commit bytes')
            self.catalog[slot]=digest
            self.event('commit',slot=slot,digest=digest,source=source,units=1)
            return 200,dict(committed=slot,digest=digest)
        if op=='activate':
            slot=body['slot']; need(slot in self.catalog,'unknown activation target'); old=self.pointer;self.pointer=slot
            self.event('activate',old=old,new=slot,units=1,actor='scripted_pi_tool',request_id=f'call_p2_{self.turn}')
            return 200,dict(active=slot)
        if op=='loaded':
            old=self.installed; origin=body['origin'];digest=body['digest'];
            need((origin=='frozen' and self.frozen is not None and old is None and digest==self.frozen)
                 or (origin=='local' and digest in self.spec['variants'].values()),'unrecognized tool load')
            self.installed=(origin,digest);self.event('loaded',origin=origin,digest=digest,units=1)
            return 200,dict(loaded=True)
        if op=='invoke':
            need(self.installed==(body['origin'],body['digest']) and body['id']==f'call_p2_{self.turn}', 'tool invocation identity mismatch')
            name, args = self.spec['actions'][self.turn-1]; need(name=='count_hits' and body['args']==args,'tool input mismatch')
            n=args['items'].count('hit')
            result={'error':'TOY_ZERO_BUG'} if n==0 and body['digest']==self.spec['variants']['bad'] else {'n':n}
            need(body['result']==result,'tool result differs from installed variant')
            challenge=secrets.token_hex(16);self.expect=(body['id'],challenge)
            self.event('invoked',id=body['id'],origin=body['origin'],digest=body['digest'],input_sha=sha(enc(args)),result=result,challenge_sha=sha(challenge),units=1)
            return 200,dict(challenge=challenge)
        if op=='observation':
            what=body['op']
            if what=='direct_frozen_write': need(body['code'] in ('EROFS','EACCES','EPERM'),'frozen write was not rejected')
            elif what in ('make_private','check_private'):
                keys={'script','cache','log','session','notes'};values=body['values'];need(set(values)==keys,'missing private spaces')
                need(all(x==sha(self.spec['marker']+':'+k) if what=='make_private' else x is None for k,x in values.items()), 'private marker invariant')
            elif what=='protected_check': need(body['values']==dict.fromkeys(['ledger','tool','receipts','sealed'],'ENOENT'),'protected filesystem visible')
            else: raise ValueError('unknown observation')
            self.event('observation',observation=body,units=1)
            return 200,dict(observed=what)
        if op=='develop':
            engine=self.develop();need(sha(engine)==self.spec['common']['engine_sha'],'development engine drift')
            result='valid-synthetic-engine' if engine.startswith(b'toy engine 0,') else 'invalid-engine'
            self.event('develop',engine_sha=sha(engine),result=result,units=3)
            return 200,dict(result=result)
        self.event('denied',operation='unregistered',units=1)
        return 404,dict(denied='unregistered')

    def model(self,b,auth):
        need(auth=='Bearer toy-only','non-fixture model credential'); need(not self.complete, 'request after complete')
        self.turn+=1; turn=self.turn; actions=self.spec['actions'];need(turn<=len(actions)+1<=self.spec['common']['turn_cap'],'turn limit')
        need(b.get('model')=='deepseek-flash' and b.get('thinking')=={'type':'enabled'} and b.get('reasoning_effort')=='max'
             and b.get('max_tokens')==self.spec['common']['output_cap'] and b.get('stream') is True,'model configuration mismatch')
        tools={item['function']['name']:item for item in b['tools']}; need(set(tools)==set(TOOLS)|({'count_hits'} if self.installed else set()),'tool registration mismatch')
        msgs=b['messages']; systems=[m['content'] for m in msgs if m['role']=='system']; users=[m['content'] for m in msgs if m['role']=='user'];
        prior=None
        if turn==1:
            need(self.spec['marker'] not in enc(msgs).decode(),'private marker flowed in first model request')
        else:
            prev_name,prev_args=actions[turn-2]; cid=f'call_p2_{turn-1}'
            response=[m for m in msgs if m.get('tool_call_id')==cid];assistant=[m for m in msgs if m['role']=='assistant']
            need(len(response)==1 and any(m.get('reasoning_content')==REASON and any(t.get('id')==cid for t in m.get('tool_calls',[])) for m in assistant),'prior reasoning/tool response missing')
            payload=enc(response[0]['content']).decode();prior=prev_name
            if self.expect is not None:
                need(self.expect[0]==cid and self.expect[1] in payload,'external challenge did not reach model')
                self.expect=None
            elif prev_name=='count_hits':
                need(self.role=='B' and turn==4 and ('not found' in payload.lower() or 'unknown' in payload.lower() or 'not available' in payload.lower()),'unknown tool was not refused: '+payload[:200])
                self.event('unavailable',tool=prev_name,request_id=cid,units=1)
            if prev_name=='read':
                need(self.spec['fact'] in payload,'facts mismatch')
                self.event('read_fact',digest=sha(self.spec['fact']),units=1)
        self.event('model',turn=turn,model='deepseek-flash',thinking='enabled',effort='max',output_cap=b['max_tokens'],
                   tools=sorted(tools),common_tools_sha=sha(enc({k:tools[k] for k in TOOLS})),system_sha=sha(enc(systems)),user_sha=sha(enc(users)),previous=prior,units=1,
                   mock_usage=dict(input=60,output_including_reasoning=12,reasoning=5))
        self.complete=turn==len(actions)+1
        if self.complete: self.event('complete',catalog=self.catalog,pointer=self.pointer,turns=turn,protected_after=self.protected_hashes())
        return 200,chunks(turn,actions[turn-1] if not self.complete else None)


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*unused): pass
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'));need(0<n<=524288,'request size');body=json.loads(self.rfile.read(n));need(isinstance(body,dict),'invalid payload')
            status,value=self.server.boundary.handle(self.path,body,self.headers.get('Authorization'))
        except (ValueError,KeyError,TypeError) as e:
            with self.server.boundary.lock:
                self.server.boundary.event('protocol_failure',error=str(e)[:300],units=1)
            status,value=400,dict(error='fixture boundary rejected request')
        raw=value if isinstance(value,bytes) else enc(value)
        self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','text/event-stream' if isinstance(value,bytes) else 'application/json');self.end_headers();self.wfile.write(raw)


def main():
    spec=json.loads(Path('/input/broker.json').read_text())
    # This trusted root is deliberately not mounted in the guest.
    trusted=Path('/trusted');trusted.mkdir()
    for name in ('ledger','tool','receipts','sealed'):
        (trusted/name).write_text(secrets.token_hex(32))
    def trusted_hashes(): return {p.name:sha(p.read_bytes()) for p in trusted.iterdir()}
    initial=trusted_hashes()
    class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
        daemon_threads=True
    server=Server('/gate/gateway.sock',Handler)
    os.chmod('/gate/gateway.sock',0o666)
    emit=lambda x:print(enc(x).decode(),flush=True)
    server.boundary=Boundary(spec,emit,trusted_hashes,lambda:Path('/input/engine.txt').read_bytes())
    server.boundary.event('ready',common=spec['common'],frozen=spec['frozen'],version_id=spec['version_id'],protected_before=initial)
    server.serve_forever()


if __name__=='__main__': main()
