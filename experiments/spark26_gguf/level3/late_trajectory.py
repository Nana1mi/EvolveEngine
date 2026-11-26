"""Shadow late-decode distribution probe. Does not replace the live v2 Judge.

Capture survivor probabilities in the real stream only at the registered tail
positions. Evaluate against an independent full-vocabulary E0 prefill on each
actual prefix. Missing mass bounds the evidence rather than becoming a zero.
"""
from math import exp, fsum

from experiments.spark26_gguf.level3 import calibrate_isolation_cache as cold, partial_tv
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical, digest

TOP_K=4096
NORMALIZATION_SLACK=.001  # existing full-vocabulary logprob parser's mass precision
TV_LIMIT=.05               # existing development full-vocabulary absolute boundary


def instrument(cold_source):
    patches=(
      (b'def stream(ids, cap, gate, *, survivor, trigger=None, call=post):\n',
       b'TAIL_TOP_K = '+str(TOP_K).encode()+b'\n\ndef stream(ids, cap, gate, *, survivor, trigger=None, call=post):\n    tail_top={}\n'),
      (b'token_ids_logprob=list(q.MARKERS),logprob_start_len=-1)',
       b'token_ids_logprob=list(q.MARKERS),logprob_start_len=-1,top_logprobs_num=TAIL_TOP_K)'),
      (b"                vectors=m['output_token_ids_logprobs']; emitted=m['output_token_logprobs']\n",
       b"                vectors=m['output_token_ids_logprobs']; emitted=m['output_token_logprobs']\n"
       b"                top=m['output_top_logprobs']\n"
       b"                q.need(type(top) is list and len(top) in (n,n-before), 'missing bound tail probabilities')\n"
       b"                for step in q.TAIL:\n"
       b"                    if before<=step<n:\n"
       b"                        row=top[len(top)-(n-step)]\n"
       b"                        q.need(type(row) is list and len(row)==TAIL_TOP_K, 'wrong width for late-decode probabilities')\n"
       b"                        tail_top[str(step)]=[[sc,tok] for sc,tok,*_ in row]\n"),
      (b'done=done,progress=progress,probs=probs)',b'done=done,progress=progress,probs=probs,tail_top=tail_top)'),
      (b"        w,ids,values=generate(job['ids'],1,job['wanted'])\n"
       b"        rows.append(dict(w,key=job['key'],generated=ids[0],**{'vector' if job['wanted']=='all' else 'values':values[0]}))\n",
       b"        tail=job['key'].startswith('i/')\n"
       b"        w,ids,values=generate(job['ids'],1,'all' if tail else job['wanted'])\n"
       b"        data={'vector':values[0]}\n"
       b"        if tail:\n"
       b"            vector=q.unpack(values[0]); data.update(values=[vector[t] for t in job['wanted']])\n"
       b"        rows.append(dict(w,key=job['key'],generated=ids[0],**data))\n"),
    )
    for left,right in patches:
        q.need(cold_source.count(left)==1,'tail adapter cannot bind the pinned cold observer')
        cold_source=cold_source.replace(left,right,1)
    return cold_source


def view(survivor, step):
    selected={}
    frames=survivor['tail_top']
    q.need(type(frames) is dict and set(frames)=={str(i) for i in q.TAIL},'missing late-decode positions')
    top=frames[str(step)]
    q.need(type(top) is list and len(top)==TOP_K,'missing top-probability coordinates')
    for pair in top:
        q.need(type(pair) is list and len(pair)==2,'wrong late-decode coordinate')
        lp,t=pair
        q.need(q.token(t) and q.lp(lp) and t not in selected,'invalid or repeated late-decode coordinate')
        selected[t]=lp
    # Require consistent evidence when a coordinate is available via both
    # endpoints of the same response. This is not a tolerance between engines.
    values=survivor['probs'][step]; tokens=(*q.MARKERS,survivor['ids'][step]); required={}
    for t,lp in zip(tokens,values,strict=True):
        q.need(q.token(t) and q.lp(lp) and (t not in required or required[t]==lp)
               and (t not in selected or selected[t]==lp),'inconsistent selected vs top logprob')
        required[t]=lp
    return selected|required


def adjusted_norm(left, right, ref_l, ref_r):
    ids=left.keys() & right.keys()
    q.need(bool(ids) and len(ref_l)==len(ref_r)==q.VOCAB,'mismatched paired support')
    px,qx=(fsum(exp(v[t]) for t in sorted(ids)) for v in (left,right))
    remain_l=max(0.,1+NORMALIZATION_SLACK-px); remain_r=max(0.,1+NORMALIZATION_SLACK-qx)
    hit=[]; miss=[]
    for t,(a,b) in enumerate(zip(ref_l,ref_r,strict=True)):
        shift=exp(b)-exp(a)
        if t in ids: hit.append(abs((exp(right[t])-exp(left[t]))-shift))
        else: miss.append(abs(shift))
    # On unmatched coordinates triangle inequality supplies an upper bound,
    # even when cancellation and completion follow different legal prefixes.
    half=fsum(hit)/2
    return dict(lower=half,upper=half+(remain_l+remain_r+fsum(miss))/2,shared=len(ids),
                observed_mass=[px,qx],norm='half_L1_of_difference_of_reference_adjusted_arms')


def tail_report(candidate,teacher, *, require_e0_ceiling=True):
    old=cold.compare(candidate,teacher,require_e0_ceiling=require_e0_ceiling)
    jobs={x['key']:x for x in q.teacher_plan(candidate)}
    rows={x['key']:x for x in teacher['rows']}; points=[]
    for pair in candidate['isolation']:
        k=pair['trigger']
        for step in q.TAIL:
            views=[]; refs=[]; result=dict(trigger=k,step=step,arms={})
            for arm in ('complete','cancel'):
                key=f'i/{k}/{arm}/{step}'; actual=view(pair[arm][0],step); row=rows[key]; ref=q.unpack(row['vector'])
                q.need(row['values']==[ref[t] for t in jobs[key]['wanted']],'teacher selected view changed')
                bound=partial_tv.matched_tv(actual,{t:ref[t] for t in actual},q.VOCAB,normalization_error=NORMALIZATION_SLACK)
                result['arms'][arm]=dict(tv=bound,status=partial_tv.certify(bound,TV_LIMIT),
                                        eos_log_delta=actual[q.EOS]-ref[q.EOS])
                views.append(actual); refs.append(ref)
            result['paired']=adjusted_norm(*views,*refs)
            points.append(result)
    states={r['status'] for point in points for r in point['arms'].values()}
    combined='different' if 'different' in states else 'inconclusive' if 'inconclusive' in states else 'certified'
    return dict(legacy_observation=old,tail_points=points,absolute_tail_tv=combined,absolute_tail_tv_limit=TV_LIMIT,
                live_gate=False)


class TailLane(cold.ColdLane):
    def __init__(self, quality, paths, helper):
        super().__init__(quality,paths,helper)
        self.cold_code=self.code
        self.files['observe_trajectory.py']=instrument(self.files['observe_trajectory.py'])
        self.code=digest(canonical({k:digest(v) for k,v in self.files.items()}))
