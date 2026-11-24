"""Quality only: native service on container loopback; synthetic text and token IDs."""
import array, base64, hashlib, json, math, sys, zlib
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
from transformers import AutoTokenizer

LABEL=sys.argv[1]; BASE='http://127.0.0.1:30091'; MODEL=Path('/model'); VOCAB=131072

def emit(kind,**r): print(json.dumps(dict(kind=kind,**r),separators=(',',':')),flush=True)
def require(ok,reason):
    if not ok: raise ValueError(reason)
def call(route,p):
    req=Request(BASE+route,json.dumps(p).encode(),{'Content-Type':'application/json'});return build_opener(ProxyHandler({})).open(req,timeout=70)
def run_case(name,fn):
    try: emit(name,ok=True,**fn())
    except Exception as e: emit(name,ok=False,error_type=type(e).__name__,detail=str(e)[:230])
def natural(rid,ids):
    with call('/generate',dict(rid=rid,input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=48))) as r: data=json.load(r)
    m=data['meta_info']; require(m['id']==rid and m['prompt_tokens']==len(ids),'response identity or prompt length');return dict(rid=rid,input_sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),ids=data['output_ids'],n=m['completion_tokens'],finish=m['finish_reason'],cap=48)
def abort():
    rid='g1-abort-'+LABEL+'-target'; seen=0; trigger=None; status=None; finish=None; done=False; last_n=None; error=None
    with call('/generate',dict(rid=rid,input_ids=[6000+i%31 for i in range(260)],stream=True,sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=512))) as r:
        for line in r:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]': done=True;break
            msg=json.loads(data)
            if 'error' in msg:
                error=msg['error'];continue
            m=msg['meta_info'];require(m['id']==rid,'aborted request identity');seen=m['completion_tokens'];last_n=seen
            if m.get('finish_reason'):finish=m['finish_reason']
            if seen>=4 and trigger is None and finish is None:
                trigger=seen
                with call('/abort_request',{'rid':rid,'abort_all':False}) as a: status=a.status; a.read()
    return dict(rid=rid,trigger=trigger,abort_http=status,finish=finish,done=done,last_n=last_n,cap=512,stream_error=error)
def numeric(n):
    ids=[6510+i%37 for i in range(n)]; rid=f'g1-{LABEL}-numeric-{n}';
    with call('/generate',dict(rid=rid,input_ids=ids,return_logprob=True,token_ids_logprob=list(range(VOCAB)),logprob_start_len=-1,sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True))) as r: data=json.load(r)
    m=data['meta_info'];require(m['id']==rid and m['prompt_tokens']==n and m['completion_tokens']==1 and m['finish_reason']['type']=='length','numeric request identity'); steps=m['output_token_ids_logprobs'];require(len(steps)==1 and len(steps[0])==VOCAB,'incomplete vocab'); vals=array.array('f',[0])*VOCAB;seen=set()
    for row in steps[0]:
        lp,i=row[:2]; require(type(i)is int and 0<=i<VOCAB and i not in seen and math.isfinite(lp),'invalid id or value'); vals[i]=lp;seen.add(i)
    if sys.byteorder!='little':vals.byteswap()
    b=vals.tobytes();return dict(count=n,prefix_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),vocab=VOCAB,format='little-f32-zlib-base64',sha256=hashlib.sha256(b).hexdigest(),vector=base64.b64encode(zlib.compress(b)).decode())

cfg=json.loads((MODEL/'config.json').read_text());require(cfg['vocab_size']==VOCAB and cfg['eos_token_id']==1,'wrong model semantics');tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True); tpl=(MODEL/'chat_template.jinja').read_text();prompts=['Reply with the single word READY.','请只回复这两个汉字：就绪。不要加引号或解释。']; inputs=[tok.encode(tok.apply_chat_template([{'role':'user','content':t}],chat_template=tpl,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for t in prompts]
emit('identity',label=LABEL,eos=cfg['eos_token_id'],vocab=VOCAB)
for i,ids in enumerate(inputs): run_case('before'+str(i),lambda: natural(f'g1-{LABEL}-before-{i}',ids))
run_case('cancel',abort)
for i,ids in enumerate(inputs): run_case('after'+str(i),lambda: natural(f'g1-{LABEL}-after-{i}',ids))
for n in (130,260): run_case('numeric'+str(n),lambda: numeric(n))
