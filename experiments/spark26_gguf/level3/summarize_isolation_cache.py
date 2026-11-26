"""Verify and summarize a completed cold E0 matrix without issuing a live threshold."""
import argparse
from math import isfinite
from pathlib import Path

from experiments.spark26_gguf.level3 import calibrate_isolation_cache as cold
from experiments.spark26_gguf.level3 import control_run as c
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256

METRICS={'abs_marker_probability','abs_draw_probability','abs_eos_logp',
         'paired_marker_probability','paired_draw_probability','paired_eos_logp'}


def summary(frames):
    q.need(len(frames)==len(cold.ORDER),'cold E0 matrix incomplete')
    groups={}
    for row,(mode,trial,teacher) in zip(frames,cold.ORDER,strict=True):
        q.need(type(row) is dict and (row['configuration'],row['candidate'],row['teacher'])==(mode,trial,teacher),
               'wrong independent E0 matrix order')
        metrics=row['metrics']; v2=metrics['reference']
        q.need(v2['protocol']==q.VERSION and set(v2['maxima'])==q.METRICS and
               type(v2['maxima']['decode_tv']) in (float,int) and 0<=v2['maxima']['decode_tv']<=.05,
               'missing or out-of-bounds E0 Decode evidence')
        cold.verify_cache({'cache_parity':metrics['cache']})
        points=metrics['points']; wanted=[(k,i) for k in q.TRIGGERS for i in q.TAIL]
        q.need(type(points) is list and len(points)==len(wanted),'wrong late-decode point count')
        group=groups.setdefault(mode,dict(arms=[],coverage_by_metric={x:0 for x in sorted(METRICS)},
                                          maxima={x:None for x in sorted(METRICS)}))
        group['arms'].append(dict(candidate=trial,teacher=teacher,node_ms=row['node_ms'],
                                  decode_tv=v2['maxima']['decode_tv']))
        for p,(k,i) in zip(points,wanted,strict=True):
            q.need(set(p)==METRICS|{'trigger','step','same_prefix','same_draw'} and (p['trigger'],p['step'])==(k,i) and
                   all(type(p[v]) is bool for v in ('same_draw','same_prefix')) and
                   ((p['paired_draw_probability'] is not None)==p['same_draw']),
                   'unbound late-decode coordinates or drawn-token witness')
            for key in METRICS:
                v=p[key]
                if key=='paired_draw_probability' and v is None: continue
                q.need(type(v) in (int,float) and isfinite(v) and v>=0,'invalid measured E0 residual')
                prior=group['maxima'][key]
                group['maxima'][key]=v if prior is None else max(prior,v)
                group['coverage_by_metric'][key]+=1
    return groups


def run(root,result_pin,output):
    res=c.read_pinned(root/'result.json',result_pin)
    plan=c.read_pinned(root/'frozen-plan.json',res['plan'])
    q.need(res['schema']==plan['schema']==cold.SCHEMA and res['status']=='E0_observations_only' and
           plan['scope']=='E0_only_non_scoring_no_admission' and
           res['gate_active'] is False and res['holdout'] is plan['holdout'] is plan['retry'] is False and
           res['paid_calls']==plan['paid_calls']==0 and
           plan['order']==[list(x) for x in cold.ORDER] and
           len(set(plan['deployments'].values()))==len(plan['deployments'])==2 and
           plan['driver']==file_sha256(Path(cold.__file__)) and
           plan['parity_helper']==file_sha256(Path(cold.__file__).with_name('isolation_cache_parity.py')) and
           len(res['runs'])==len(cold.ORDER),'not a complete frozen engineering matrix')
    observed=[]; distinct=set()
    for info,(mode,trial,teacher) in zip(res['runs'],cold.ORDER,strict=True):
        row=c.read_pinned(root/(mode+'-'+trial+'-metrics.json'),info['sha256'])
        q.need((info['configuration'],info['candidate'],info['teacher'])==(mode,trial,teacher) and
               row['plan']==res['plan'] and type(row['node_ms']) is int and row['node_ms']>0 and
               c._sha(row['candidate_pin']) and c._sha(row['teacher_pin']), 'matrix receipt not bound to plan')
        distinct.update((row['candidate_pin'],row['teacher_pin']))
        observed.append(row)
    q.need(len(distinct)==2*len(cold.ORDER),'non-independent service receipts reused')
    return write(output,dict(status='descriptive_E0_only_no_gate',result=result_pin,plan=res['plan'],
       cold_configurations=summary(observed),holdout=False,paid_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('result_pin');p.add_argument('new_summary',type=Path);a=p.parse_args()
    print(run(a.directory,a.result_pin,a.new_summary))
