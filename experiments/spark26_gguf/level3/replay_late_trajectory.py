"""Replay bound late-decode E0 evidence and validate the split on altered copies.

No server, compilation, paid requests, or scoring mutation. Faults are injected
into copied observer probabilities, not into E0 or an Agent's source tree.
"""
import argparse
from copy import deepcopy
from math import exp, expm1, log1p
from pathlib import Path

from experiments.spark26_gguf.level3 import control_run as c, late_trajectory as late, split_quality as score
from experiments.spark26_gguf.level3.analyze_isolation_residual import pinned_json
from experiments.spark26_gguf.level3.calibrate_late_trajectory import SCHEMA
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256


def rewrite(survivor, step, token, delta):
    view=late.view(survivor,step); correction=log1p(exp(view[token])*expm1(delta))
    def tilt(v,t): return min(0.,v+(delta if t==token else 0.)-correction)
    probe=survivor['tail_top'][str(step)]
    survivor['tail_top'][str(step)]=[[tilt(v,t),t] for v,t in probe]
    survivor['probs'][step]=[tilt(v,t) for v,t in zip(survivor['probs'][step],(*q.MARKERS,survivor['ids'][step]),strict=True)]


def check_controls(candidate, teacher, report, limits):
    baseline=score.assess(report,limits)
    if baseline['status']!='valid':
        return dict(baseline=baseline,synthetic_controls='not_run_without_valid_e0')
    old=canonical(candidate); checks={}; step=q.TAIL[0]
    cap=limits['decode_eos_logp']
    q.need(cap>0,'empty EOS control tolerance')
    left,right=(report['tail_points'][0]['arms'][arm]['eos_log_delta'] for arm in ('complete','cancel'))
    # Select signed control magnitudes from the *already registered* limits and
    # clean witness, not from any previously failed or successful candidate.
    fixtures={
       'common_eos': ({'complete':cap+1.-left,'cancel':cap+1.-right},'mixed_tail_absolute'),
       'paired_eos': ({'complete':-.75*cap-left,'cancel':.75*cap-right},'paired_cancel_numeric'),
    }
    for label,(angles,expected) in fixtures.items():
        altered=deepcopy(candidate)
        for arm,delta in angles.items(): rewrite(altered['isolation'][0][arm][0],step,q.EOS,delta)
        result=score.assess(late.tail_report(altered,teacher),limits)
        groups={k:v['status'] for k,v in result['groups'].items()}
        # The named group must actually reject the fault. Any unexpected
        # secondary flag is preserved; it is not reassigned to the intended group.
        expected_key='paired_eos' if label=='paired_eos' else '/eos'
        q.need(groups[expected]=='violation' and any(expected_key in s for s in result['groups'][expected]['violations']),
               'split quality control escaped: '+label)
        checks[label]=result
    q.need(old==canonical(candidate),'fault replay mutated the original observer receipt')
    return dict(baseline=baseline,synthetic_controls=checks)


def run(root, pin, primary, out):
    result=c.read_pinned(root/'result.json',pin,2<<20); plan=c.read_pinned(root/'frozen-plan.json',result['plan'])
    q.need(plan['schema']==result['schema']==SCHEMA and result['scope']=='one_e0_feasibility_no_live_gate' and
           plan['scope']=='E0_observation_only' and plan['holdout'] is result['holdout'] is plan['retry'] is False and
           plan['paid_calls']==result['paid_calls']==0 and plan['target']==late.TV_LIMIT and plan['top_k']==late.TOP_K and
           plan['metric']==file_sha256(Path(late.__file__)) and plan['bounds']==file_sha256(Path(late.partial_tv.__file__)),
           'expected frozen late E0 feasibility evidence')
    accepted=c.read_pinned(primary/'controls/acceptance.json',plan['acceptance'])
    registered=c.read_pinned(primary/'trajectory/calibration.json',accepted['trajectory'])
    q.need(registered['limits']==q.envelope([r['metrics'] for r in registered['arms']]) and
           accepted['registration']==registered['registration_sha256']==plan['registration'] and
           set(registered['deployments'].values())=={plan['candidate'],plan['teacher']},'wrong independent E0 calibration')
    raw={}; bindings={}; common_weights=None
    for role in ('candidate','teacher'):
        directory=root/('trajectory-'+role)
        proof=c.read_pinned(directory/'trajectory.json',result[role+'_receipt'])
        q.need(proof['role']==role and proof['protocol']==q.VERSION and proof['code']==plan['observer'] and
               proof['deployment']==plan[role] and proof['binding']['origin']=='hardware_collection' and
               proof['binding']['engine']==registered['e0'],'misbound late hardware receipt')
        files=proof['files']; must={'observer.stdout','observer.stderr','observer.inspect.json','external-identity.json',
            'service.inspect.json','host-mappings.json','inputs/trajectory-plan.json','inputs/observe_trajectory.py',
            'inputs/trajectory_contract.py','inputs/observe_timing.py','inputs/timing.py','inputs/observe_text_timing.py',
            'inputs/isolation_cache_parity.py'}
        q.need(set(files)==must and all((directory/k).resolve(strict=True)==directory/k and file_sha256(directory/k)==v
                                       for k,v in files.items()),'late hardware witness changed')
        sources={name.removeprefix('inputs/'):files[name] for name in files if name.startswith('inputs/') and name.endswith('.py')}
        q.need(digest(canonical(sources))==plan['observer'],'late observer sources changed')
        raw[role]=pinned_json(directory/'observer.stdout',files['observer.stdout'])
        request=dict(protocol=q.VERSION,role=role)
        if role=='teacher': request['jobs']=q.teacher_plan(raw['candidate'])
        q.need((directory/'inputs/trajectory-plan.json').read_bytes()==canonical(request),'late observer request changed')
        bindings[role]=proof['binding']
        if common_weights is not None: q.need(proof['weights']==common_weights,'teacher weights changed')
        common_weights=proof['weights']
    a,b=bindings['candidate'],bindings['teacher']
    q.need(a['service_id']!=b['service_id'] and a['run_id']!=b['run_id'] and a['end_ns']<=b['start_ns'] and
           all(a[k]==b[k] for k in ('worker_id','boot_id','device_uuid')),'late evidence not independent on the same registered device')
    replay=late.tail_report(raw['candidate'],raw['teacher'])
    q.need(replay==result['observe'],'late E0 observation changed after replay')
    controls=check_controls(raw['candidate'],raw['teacher'],replay,registered['limits'])
    return write(out,dict(scope='offline_e0_feasibility_and_fault_validation_not_live_admission',source=pin,plan=result['plan'],
       trajectory_v2=accepted['trajectory'],checker=file_sha256(Path(__file__)),metric_rule=file_sha256(Path(score.__file__)),
       checks=controls,paid_calls=0,holdout=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); p.add_argument('pin')
    p.add_argument('primary',type=Path);p.add_argument('output',type=Path); args=p.parse_args()
    print(run(args.root,args.pin,args.primary,args.output))
