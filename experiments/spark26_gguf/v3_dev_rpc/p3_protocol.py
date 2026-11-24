"""Predeclared P3 exploration reference, handoff validation, and W cost proposal."""
from math import ceil, exp, isfinite, log
from pathlib import Path
import re

from evolveengine.v3_harness import canonical, digest
from node_client import need

SCENES=('W1','W2','W3','W4')
OPS=('develop_quality','develop_measure')
STATUSES={'valid','violation','insufficient_evidence','observed'}
HEX=re.compile('[0-9a-f]{64}')


def metrics(scenes):
    need(type(scenes) is dict and set(scenes)==set(SCENES),'four scene metrics required')
    scores={k:scenes[k]['score'] for k in SCENES}
    need(all(type(v) in (int,float) and isfinite(v) and v>0 for v in scores.values()),'invalid scene metrics')
    return scores


def exploratory_index(scenes,e0):
    candidate=metrics(scenes);reference=metrics(e0)
    ratios={k:candidate[k]/reference[k] for k in SCENES}
    return dict(index=exp(sum(log(v) for v in ratios.values())/4),ratios=ratios,e0=reference,
                scope='one fresh E0 boot and one candidate boot in the same episode; directional only, no noise interval or speedup claim')


def source_identity(sources):
    return digest(canonical({n:digest(x) for n,x in sources.items()}))


def validate_handoff(s1, start_bytes, saved_bytes, expected_h, replay):
    """Verify serialized S1 output, trusted ledgers and (in live mode) every raw DEV archive."""
    need(s1['arm']=='S1' and s1['status']=='completed' and s1['next_H']==expected_h,'S1 state not settled for handoff')
    need(s1['runner']['status']=='pi_session_complete' and s1['runner']['isolated_state_removed'], 'S1 sandbox not settled')
    need(s1['start_E']==source_identity(start_bytes) and s1['end_E']==source_identity(saved_bytes),'S1 engine artifact hash mismatch')
    baseline=s1['calibration'];need(baseline['operation']=='develop_measure' and baseline['status']=='observed'
         and baseline['source_sha']==digest(next(iter(start_bytes.values()))) and baseline['collocated_e0'] is True,
         'no verified S1 same-session E0')
    metrics(baseline['summary']['scenes']);replay(baseline)
    rows=s1['rows'];need([row['event_id'] for row in rows]==list(range(1,len(rows)+1)),'invalid S1 event chronology')
    logs=[x for x in rows if x['operation'] in OPS and x.get('receipt_id')]
    jobs=s1['workers'];need(len(logs)==len(jobs)<=4,'DEV facts incomplete')
    facts_rows=[]
    for job,event in zip(jobs,logs,strict=True):
        op=job['operation'];status=job['status'];rid=job['receipt_id'];eng=job['engine'];cost=job['node_ms']
        need(op in OPS and status in STATUSES and HEX.fullmatch(rid) and HEX.fullmatch(eng) and HEX.fullmatch(job['source_sha']) and
             type(cost) is int and cost>0 and all(job[k]==event[k] for k in ('operation','status','engine','receipt_id')) and event['node_ms_charged']==cost,
             'DEV receipt does not match trusted event')
        replay(job)
        out=dict(operation=op,status=status,receipt_id=rid,node_ms=cost,engine=eng)
        data=job['summary']
        if op=='develop_measure' and status=='observed':
            out['scores']=metrics(data['scenes']);out['index_vs_S1_E0']=exploratory_index(data['scenes'],baseline['summary']['scenes'])['index']
        if 'max_tv' in data:
            tv=data['max_tv'];eos=data['eos_passed'];kv=data['mixed_passed']
            need(type(tv) in (float,int) and isfinite(tv) and 0<=tv<=1 and type(eos) is int and 0<=eos<=4 and type(kv) is bool, 'invalid DEV quality facts')
            out.update(max_tv=tv,eos_passed=eos,mixed_passed=kv)
        facts_rows.append(out)
    need(s1['node_ms']==baseline['node_ms']+sum(x['node_ms'] for x in jobs), 'S1 cost does not reconcile')
    selections=[x for x in rows if x['operation']=='deliver_engine' and x['status']=='selected_by_agent']
    if not selections:
        need(s1['explicitly_delivered'] is None and saved_bytes==start_bytes,'unsigned source at S2 ingress')
    else:
        s=selections[-1]
        need(s1['explicitly_delivered']==s['engine'] and any(e['operation']=='develop_quality' and e['status']=='valid' and
             (e['receipt_id'],e['engine'],e['source_sha'])==(s['receipt_id'],s['engine'],digest(next(iter(saved_bytes.values())))) for e in jobs),
             'delivered source lacks matching positive receipt')
    active=[x for x in rows if x['operation']=='harness_activate' and x['status']=='active']
    if not active:need(s1['start_H']==s1['next_H'],'unregistered Harness transfer')
    else:need(active[-1]['new']==s1['next_H'],'Harness pointer is not last explicit activation')
    result=dict(schema='v3-p3-s1-facts-v1',prior_arm='S1',engine_delivered=bool(selections),harness_changed=s1['start_H']!=s1['next_H'],
                e0_calibration=dict(receipt_id=baseline['receipt_id'],scores=metrics(baseline['summary']['scenes'])),observations=facts_rows)
    need(0<len(canonical(result))<=24000,'invalid S2 handoff size');return result


# Units here and in the final record are seconds, tokens and calls (not a cost or power estimate).
FLOORS=dict(wall_seconds=180,gpu_node_seconds=200,input_tokens=10000,output_tokens=2000,development_calls=1)
GRID=dict(wall_seconds=60,gpu_node_seconds=60,input_tokens=1000,output_tokens=1000,development_calls=1)


def propose_w(sessions):
    if len(sessions)!=2 or any(x['status']!='completed' or x.get('budget_stop') for x in sessions):
        return dict(status='hold_requires_settled_uncensored_costs',w_started=False)
    actual=[dict(wall_seconds=x['wall_ms']/1000,gpu_node_seconds=x['node_ms']/1000,input_tokens=x['usage']['input'],
                 output_tokens=x['usage']['output_including_reasoning'],development_calls=len(x['workers'])) for x in sessions]
    cap={k:GRID[k]*ceil(12.5*max(FLOORS[k],*(x[k] for x in actual))/GRID[k]) for k in FLOORS}
    return dict(status='cost_proposal_not_frozen',maximum_episodes=10,observed=actual,training_total=cap,w_started=False)
