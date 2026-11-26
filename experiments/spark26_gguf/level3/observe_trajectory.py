"""Runs inside the read-only *observer*, never inside the candidate or inside Pi."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
from time import monotonic_ns
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

import trajectory_contract as q
from observe_text_timing import merge_stream_output
from observe_timing import wait_ready


def post(route, payload):
    req=Request('http://127.0.0.1:30091/'+route,json.dumps(payload).encode(),{'Content-Type':'application/json'})
    return build_opener(ProxyHandler({})).open(req,timeout=90)


def selected(pairs, wanted):
    q.need(type(pairs) is list and len(pairs)==len(wanted),'incomplete selected logprobs')
    by_id={}
    for value,t,*_ in pairs:
        q.need(q.token(t) and t not in by_id and q.lp(value),'invalid selected probabilities')
        by_id[t]=value
    q.need(by_id.keys()==set(wanted),'wrong selected token IDs')
    return [by_id[t] for t in wanted]


def generate(ids, cap, wanted, *, call=post):
    rid=uuid4().hex
    values=list(range(q.VOCAB)) if wanted=='all' else wanted
    msg=dict(rid=rid,input_ids=ids,return_logprob=True,token_ids_logprob=values,logprob_start_len=-1,
             sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=cap))
    with call('generate',msg) as response:
        q.need(response.status==200,'missing numeric response')
        raw=response.read(64*(1<<20)+1)
        q.need(len(raw)<=64*(1<<20),'oversized numeric response')
        row=json.loads(raw)
    meta=row['meta_info']; output=row['output_ids']; steps=meta['output_token_ids_logprobs']
    q.need(meta['id']==rid and meta['prompt_tokens']==len(ids) and meta['completion_tokens']==cap==len(output)==len(steps)
           and all(map(q.token,output)) and meta['finish_reason']['type']=='length','wrong numeric request or coverage')
    witness=dict(rid=rid,prompt_sha=q.ih(ids),input=len(ids),cap=cap,finish='length')
    if wanted=='all': result=[q.pack(step) for step in steps]
    else: result=[selected(step,wanted) for step in steps]
    return witness,output,result


def stream(ids, cap, gate, *, survivor, trigger=None, call=post):
    rid=uuid4().hex; progress=[]; seq=[]; probs=[]; finish=ack=None; done=False
    params=dict(temperature=0,max_new_tokens=cap,ignore_eos=survivor)
    payload=dict(rid=rid,input_ids=ids,stream=True,sampling_params=params)
    if survivor: payload.update(return_logprob=True,token_ids_logprob=list(q.MARKERS),logprob_start_len=-1)
    gate.wait(timeout=15)
    with call('generate',payload) as response:
        q.need(response.status==200 and response.headers.get_content_type()=='text/event-stream','invalid mixed response')
        while line:=response.readline(4*(1<<20)):
            q.need(len(line)<4*(1<<20),'oversized mixed frame')
            if not line.startswith(b'data:'): continue
            q.need(not done,'data after DONE'); body=line[5:].strip()
            if body==b'[DONE]':
                done=True
                continue
            q.need(finish is None,'data after terminal frame')
            data=json.loads(body); m=data['meta_info']; n=m['completion_tokens']; term=m.get('finish_reason')
            q.need(m['id']==rid and m['prompt_tokens']==len(ids) and type(n) is int and len(seq)<=n<=cap,
                   'mixed request binding changed')
            before=len(seq); seq=merge_stream_output(seq,data['output_ids'],n,final=term is not None)
            if survivor and n>before:
                vectors=m['output_token_ids_logprobs']; emitted=m['output_token_logprobs']
                q.need(len(vectors)==len(emitted) and len(vectors) in (n,n-before), 'missing mixed step probabilities')
                for j,(vector,draw) in enumerate(zip(vectors[-(n-before):],emitted[-(n-before):]),before):
                    q.need(draw[1]==seq[j] and q.lp(draw[0]),'mixed probability bound to wrong generated token')
                    probs.append(selected(vector,q.MARKERS)+[draw[0]])
            progress.append([n,monotonic_ns()])
            if term is not None: finish=term['type']
            if trigger is not None and ack is None and finish is None and n>=trigger:
                start=monotonic_ns()
                with call('abort_request',dict(rid=rid,abort_all=False)) as reply:
                    q.need(reply.status==200,'neighbor cancellation refused'); reply.read(4096)
                ack=[n,start,monotonic_ns(),200]
    return dict(rid=rid,prompt_sha=q.ih(ids),ids=seq,cap=cap,finish=finish,ack=ack,done=done,progress=progress,probs=probs)


def capture():
    decode=[]; isolation=[]
    for case in q.CASES:
        w,ids,vectors=generate(q.prefix(case),q.STEPS,'all')
        decode.append(dict(w,case=case,ids=ids,vectors=vectors))
    for k,trigger in enumerate(q.TRIGGERS):
        pair=dict(trigger=trigger)
        for arm in (('complete','cancel') if k%2==0 else ('cancel','complete')):
            gate=Barrier(2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                survivor=pool.submit(stream,q.prefix(7),q.SURVIVOR,gate,survivor=True)
                neighbor=pool.submit(stream,q.prefix(6),q.NEIGHBOR,gate,survivor=False,trigger=trigger if arm=='cancel' else None)
                pair[arm]=[survivor.result(),neighbor.result()]
        isolation.append(q.streams(pair,trigger))
    return dict(protocol=q.VERSION,role='candidate',decode=decode,isolation=isolation)


def teach(plan):
    rows=[]
    for job in plan:
        # Always one-token prefill at the actual candidate prefix. No free E0 rollout.
        w,ids,values=generate(job['ids'],1,job['wanted'])
        rows.append(dict(w,key=job['key'],generated=ids[0],**{'vector' if job['wanted']=='all' else 'values':values[0]}))
    return dict(protocol=q.VERSION,role='teacher',rows=rows)


def main():
    wait_ready()
    plan=json.loads(Path('/observer/trajectory-plan.json').read_bytes())
    q.need(plan['protocol']==q.VERSION and plan['role'] in ('candidate','teacher'),'unknown probe role')
    data=capture() if plan['role']=='candidate' else teach(plan['jobs'])
    print(json.dumps(data,separators=(',',':'),allow_nan=False),flush=True)


if __name__=='__main__': main()
