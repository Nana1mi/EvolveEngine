"""Native SSE, no profiler/payload logging; token timing uses cumulative counters, not frame count."""
import concurrent.futures as cf
import hashlib, json, re, sys, threading, time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

ROOT=Path(__file__).parent
LOG=Path('/tmp/ee-e0-server.log')
COUNTER=time.perf_counter_ns

def request(lane,n,out,rid,gate):
    body={'rid':rid, 'input_ids':[6510+37*lane+(i%37) for i in range(n)],'stream':True,'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':out}}
    req=Request('http://127.0.0.1:30091/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
    client=build_opener(ProxyHandler({})); gate.wait(timeout=10); start=COUNTER(); first=last=finish=done=None; prev=0; digest=None; hist=[]
    with client.open(req,timeout=100) as resp:
        if resp.status!=200 or resp.headers.get_content_type()!='text/event-stream': raise ValueError('Invalid SSE')
        for line in resp:
            if not line.startswith(b'data:'): continue
            ts=COUNTER(); data=line[5:].strip()
            if data==b'[DONE]': done=ts; break
            msg=json.loads(data); meta=msg['meta_info']; k=meta['completion_tokens']; ids=msg['output_ids']
            if meta['id']!=rid or not prev<=k<=out or meta['prompt_tokens']!=n: raise ValueError(('length/id',rid,k,prev,meta['prompt_tokens']))
            if k>prev:
                hist.append(k-prev)
                if first is None:
                    if k!=1: raise ValueError(('first token ambiguous',rid,k))
                    first=ts
                if k==out: last=ts
                prev=k
            if meta.get('finish_reason') is not None:
                if finish is not None or meta['finish_reason']['type']!='length' or len(ids)!=out or k!=out or not all(type(i) is int for i in ids): raise ValueError(('unexpected termination',rid))
                finish=ts; digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    if not(first and last and finish and done and start<first<last<=finish<=done and prev==out): raise ValueError(('incomplete',rid))
    return dict(rid=rid,input=n,output=out,sha256=digest,start_ns=start,first_ns=first,last_ns=last,done_ns=done,ttft_ms=(first-start)/1e6,e2e_ms=(done-start)/1e6,decode_tps=(out-1)*1e9/(last-first),max_stream_step=max(hist))

def main(label):
    p=json.loads((ROOT/'plan.json').read_text()); order=next(o for labels,o in p['pairs'] if label in labels)
    for sc in order:
        spec=p['scenes'][sc]; width=len(spec)
        for idx in range(p['warm_waves']+p['timed_waves']):
            off=LOG.stat().st_size; gate=threading.Barrier(width)
            with cf.ThreadPoolExecutor(max_workers=width) as ex:
                fs=[ex.submit(request,i,n,k,f'{label}-{sc}-{idx}-{i}',gate) for i,(n,k) in enumerate(spec)]; rows=[f.result() for f in fs]
            with LOG.open('rb') as log:
                log.seek(off); interval=log.read().decode(errors='replace')
            widths=list(map(int,re.findall(r'Decode batch, #running-req: (\d+)',interval)))
            dt=(max(r['done_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
            overlap=(min(r['last_ns'] for r in rows)-max(r['first_ns'] for r in rows))/1e6
            pval=(rows[0]['decode_tps'] if sc in ('W1','W2') else 1000/rows[0]['ttft_ms'] if sc=='W3' else sum(r['output'] for r in rows)/dt)
            rec=dict(kind='wave',label=label,scene=sc,index=idx,timed=idx>=p['warm_waves'],width=width,decode_widths=widths,overlap_ms=overlap,wall_s=dt,P=pval,requests=rows)
            print(json.dumps(rec,separators=(',',':')),flush=True)
            if width not in widths or (width>1 and overlap<=0): raise ValueError(('unverified active width',sc,width,widths,overlap))
if __name__=='__main__': main(sys.argv[1])
