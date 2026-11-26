"""Proposed split quality assessment; does not wire or admit the live Runner.

Same existing TV ceiling for early and late absolute correctness. Use the
original independently calibrated early EOS-log limit consistently at late
positions too, not the much smaller V2 mixed-tail scalar error as a surrogate
for cancellation. Retain the V2 cancellation-marker limit. A conservative
interval must actually certify a pass; an unknown tail is not a violation or
successful quality result.
"""
from math import isfinite

from experiments.spark26_gguf.level3 import partial_tv
from evolveengine import v3_level3_trajectory as q

VERSION='split-absolute-quality-causal-cancel-dev-v1'
TV_CEILING=.05
EARLY=('decode_tv','decode_eos_logp','decode_greedy_regret')
GROUPS=('autoregressive_full_vocab','mixed_tail_absolute','paired_cancel_numeric')


def number(x, *, signed=False):
    q.need(type(x) in (int,float) and isfinite(x) and (signed or x>=0),'invalid registered trajectory metric')
    return x


def assess(report, limits):
    """Score the proposed policy against a caller-verified historical E0 envelope.

    This pure function emits no signed receipt or launch decision. Before
    promotion, hardware controls must exercise each group independently.
    """
    q.need(set(limits)==q.METRICS,'need all legacy calibration coordinates')
    for x in limits.values(): number(x)
    q.need(limits['decode_tv']<=TV_CEILING and report['absolute_tail_tv_limit']==TV_CEILING
           and report['live_gate'] is False,'unregistered early or late TV bound')
    old=report['legacy_observation']; early=old['reference']; peaks=early['maxima']
    q.need(early['protocol']==q.VERSION and set(peaks)==q.METRICS,'missing independent early quality')
    for x in peaks.values(): number(x)
    by={key:dict(violations=[],insufficient=[]) for key in GROUPS}
    for key in EARLY:
        if peaks[key]>limits[key]: by['autoregressive_full_vocab']['violations'].append(key)
    absolute=by['mixed_tail_absolute']; paired=by['paired_cancel_numeric']
    points=report['tail_points']; old_points=old['points']; coords=[(k,i) for k in q.TRIGGERS for i in q.TAIL]
    q.need(type(points) is list and type(old_points) is list and len(points)==len(old_points)==len(coords),
           'incomplete late coverage')
    eos_cap=limits['decode_eos_logp']; distinct=set(); found_tv=set()
    def interval(group, name, frame, boundary):
        seen=partial_tv.certify(frame,boundary)
        if seen=='different': group['violations'].append(name)
        elif seen=='inconclusive': group['insufficient'].append(name)
        return seen
    for point,v2,(trigger,step) in zip(points,old_points,coords,strict=True):
        q.need((point['trigger'],point['step'])==(v2['trigger'],v2['step'])==(trigger,step) and
               (trigger,step) not in distinct and set(point['arms'])=={'complete','cancel'},'misbound late coverage')
        distinct.add((trigger,step)); tag=f'{trigger}/{step}'
        eos=[]
        for arm in ('complete','cancel'):
            item=point['arms'][arm]; delta=number(item['eos_log_delta'],signed=True); eos.append(delta)
            outcome=interval(absolute,tag+'/'+arm+'/tv',item['tv'],TV_CEILING)
            q.need(item['status']==outcome,'stored absolute TV verdict changed'); found_tv.add(outcome)
            if abs(delta)>eos_cap: absolute['violations'].append(tag+'/'+arm+'/eos')
        if abs(eos[0]-eos[1])>eos_cap: paired['violations'].append(tag+'/paired_eos')
        if number(v2['paired_marker_probability'])>limits['iso_paired_residual']:
            paired['violations'].append(tag+'/paired_marker')
        interval(paired,tag+'/paired_adjusted_norm',point['paired'],TV_CEILING)
    expected='different' if 'different' in found_tv else 'inconclusive' if 'inconclusive' in found_tv else 'certified'
    q.need(report['absolute_tail_tv']==expected,'stored overall TV verdict changed')
    def code(x):
        return 'violation' if x['violations'] else 'insufficient_evidence' if x['insufficient'] else 'valid'
    groups={k:dict(v,status=code(v)) for k,v in by.items()}
    statuses={v['status'] for v in groups.values()}
    overall=next(s for s in ('violation','insufficient_evidence','valid') if s in statuses)
    return dict(protocol=VERSION,status=overall,groups=groups,live_gate=False,
                bound_sources=dict(tv_ceiling=TV_CEILING,absolute_and_paired_eos_logp=eos_cap,
                                   paired_marker_probability=limits['iso_paired_residual']))
