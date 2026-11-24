"""Non-performance probe for predeclared, serialized DEV positions and mixed cancellation."""
import array, base64, hashlib, json, math, random, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from urllib.request import ProxyHandler, Request, build_opener

VOCAB=131072

def digest(b): return hashlib.sha256(b).hexdigest()
def prefix(i):
    n=[130,260][i%2]; rng=random.Random(0x20261123+i); alphabet=[rng.randrange(2000,28000) for _ in range(7+i%19)]
    return [rng.choice(alphabet) for _ in range(n)]

def ih(ids): return digest(json.dumps(ids,separators=(',',':')).encode())
def emit(kind,**d):print(json.dumps(dict(kind=kind,**d),separators=(',',':')),flush=True)
def call(route,p):
    req=Request('http://127.0.0.1:30091/'+route,json.dumps(p).encode(),{'Content-Type':'application/json'});return build_opener(ProxyHandler({})).open(req,timeout=40)
def params(n):return dict(temperature=0,max_new_tokens=n,ignore_eos=True)
def numeric(role,i):
    ids=prefix(i); rid=f'dev-{role}-{i:02}'; key=ih(ids); tag=Path('/tmp/ee-g1-numeric-active'); tag.write_text(rid+':'+key)
    try:
        with call('generate',dict(rid=rid,input_ids=ids,return_logprob=True,token_ids_logprob=list(range(VOCAB)),logprob_start_len=-1,sampling_params=params(1))) as r: data=json.load(r)
    finally: tag.unlink(missing_ok=True)
    m=data['meta_info']; assert m['id']==rid and m['prompt_tokens']==len(ids) and m['completion_tokens']==1 and m['finish_reason']['type']=='length'
    steps=m['output_token_ids_logprobs'];assert len(steps)==1 and len(steps[0])==VOCAB
    a=array.array('f',[0])*VOCAB; seen=set()
    for lp,t,*rest in steps[0]:
        assert type(t)is int and 0<=t<VOCAB and t not in seen and math.isfinite(lp) and lp<=0
        a[t]=lp;seen.add(t)
    if sys.byteorder!='little':a.byteswap()
    raw=a.tobytes();return dict(i=i,rid=rid,input_sha=key,n=len(ids),token=data['output_ids'][0],sha=digest(raw),vector=base64.b64encode(zlib.compress(raw)).decode())

def stream(rid,ids,cap,bar=None,trigger=None):
    if bar:bar.wait(timeout=8)
    samples=[]; ack=None;term=None;done=False; final=[]
    with call('generate',dict(rid=rid,input_ids=ids,stream=True,sampling_params=params(cap))) as r:
        for line in r:
            if not line.startswith(b'data:'):continue
            b=line[5:].strip()
            if b==b'[DONE]':done=True;break
            msg=json.loads(b);m=msg['meta_info'];assert m['id']==rid and m['prompt_tokens']==len(ids); n=m['completion_tokens'];samples.append([n,time.monotonic_ns()]); final=msg['output_ids'];term=m.get('finish_reason')
            if trigger is not None and ack is None and n>=trigger and not term:
                t=time.monotonic_ns()
                with call('abort_request',{'rid':rid,'abort_all':False}) as a: code=a.status;a.read()
                ack=[n,t,time.monotonic_ns(),code]
    return dict(rid=rid,input_sha=ih(ids),cap=cap,samples=samples,ack=ack,finish=term,done=done,ids=final)

def mixed(role,stage):
    survivor=prefix(7); r=f'devkv-{role}-{stage}';cap=160
    if not stage:return dict(survivor=stream(r+'-s',survivor,cap))
    gate=Barrier(2)
    with ThreadPoolExecutor(2) as ex:
        s=ex.submit(stream,r+'-s',survivor,cap,gate); t=ex.submit(stream,r+'-t',prefix(6),128,gate,stage)
        return dict(survivor=s.result(),target=t.result())

def main(role):
    for i in range(32):
        try:emit('numeric',ok=True,**numeric(role,i))
        except Exception as e:emit('numeric',ok=False,i=i,error=str(e)[:180],errtype=type(e).__name__)
    if role in ('e0a','e0b','b'):
        for stage in (0,8,32):
            try:emit('kv',stage=stage,ok=True,**mixed(role,stage))
            except Exception as e:emit('kv',stage=stage,ok=False,error=str(e)[:180],errtype=type(e).__name__)
if __name__=='__main__':main(sys.argv[1])
