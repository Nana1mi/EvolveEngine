"""Non-scoring DEV audit: full-vocabulary decode vs decode on identical prefixes.

Consume only already archived trusted development observer receipts. This
module is not imported by the Worker gate, scorer, or launch audit. The
calibration is descriptive, not a confidence bound or proof of quality.
"""
import argparse
import json
from math import exp, fsum
from pathlib import Path

from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256

SCHEMA='spark26-same-prefix-decode-audit-v1'
LIMITS=('tv','eos_log_delta')


def read_pin(path, pin, *, cap):
    q.need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
           0<path.stat().st_size<=cap,'unsafe or oversized development receipt')
    raw=path.read_bytes()
    q.need(digest(raw)==pin,'development receipt hash differs')
    return json.loads(raw)


def evidence(root, pin):
    row=read_pin(root/'trajectory.json',pin,cap=128<<10); binding=row['binding']
    q.need(row['role']=='candidate' and row['protocol']==q.VERSION and binding['origin']=='hardware_collection',
           'need archived hardware candidate-role decode')
    raw=read_pin(root/'observer.stdout',row['files']['observer.stdout'],cap=24<<20)
    return row, raw


def compare_decode(lhs, rhs):
    """Do not compare the first prefill position or different token prefixes."""
    for model in (lhs,rhs):
        q.need((model['role'],model['protocol'])==('candidate',q.VERSION) and len(model['decode'])==len(q.CASES),
               'missing registered decode cases')
    out=[]; coverage=[]
    for case,a,b in zip(q.CASES,lhs['decode'],rhs['decode'],strict=True):
        q.check_decode(a,case); q.check_decode(b,case)
        q.need(a['rid']!=b['rid'],'decode request was replayed as an independent service')
        full_a,full_b=([q.unpack(row) for row in engine['vectors']] for engine in (a,b))
        steps=[]
        for i in range(1,q.STEPS):
            if a['ids'][:i]!=b['ids'][:i]: break
            left,right=full_a[i],full_b[i]
            steps.append(i)
            out.append(dict(case=case,step=i,tv=fsum(abs(exp(x)-exp(y)) for x,y in zip(left,right,strict=True))/2,
                            eos_log_delta=abs(left[q.EOS]-right[q.EOS])))
        coverage.append(dict(case=case,aligned_steps=len(steps),planned_decode_steps=q.STEPS-1))
    return dict(coverage=coverage,coordinates=out,maxima={k:max(row[k] for row in out) for k in LIMITS} if out else None,
                no_matching_prefix='no_observation' if not out else None)


def compare(arm, ref, *, e0_engine, e0_deployment):
    left,lp=arm; right,rp=ref; a,b=(x['binding'] for x in (left,right))
    q.need(b['engine']==e0_engine and right['deployment']==e0_deployment and left['weights']==right['weights'] and
           a['device_uuid']==b['device_uuid'] and
           a['boot_id']==b['boot_id'] and a['service_id']!=b['service_id'] and a['run_id']!=b['run_id'],
           'reference must be a separate E0 decode on the registered weights and node')
    return compare_decode(lp,rp)


def calibrate(a,b,control, *, e0_engine, e0_deployment):
    q.need(a[0]['binding']['engine']==e0_engine and a[0]['deployment']!=e0_deployment and
           control[0]['binding']['engine']!=e0_engine,'need separate E0 builds and the legal MMVQ control')
    reports={k:compare(arm,b,e0_engine=e0_engine,e0_deployment=e0_deployment) for k,arm in
             (('e0_pair',a),('legal_mmvq_config',control))}
    q.need(all(row['maxima'] is not None for row in reports.values()), 'no common prefix: cannot calibrate audit signal')
    return dict(method='descriptive_max_of_e0_and_legal_control',comparisons=reports,
                reference_envelope={k:max(c['maxima'][k] for c in reports.values()) for k in LIMITS})


def hints(result,envelope):
    q.need(set(envelope)==set(LIMITS) and all(type(v) in (int,float) and 0<=v<float('inf') for v in envelope.values()),
           'unregistered descriptive envelope')
    if result['maxima'] is None: return dict(status='insufficient_aligned_prefix',flags=[],scoring=False)
    flags=[k for k in LIMITS if result['maxima'][k]>envelope[k]]
    return dict(status='review_suggested' if flags else 'within_observed_range',flags=flags,scoring=False)


def main():
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='mode',required=True)
    for kind in ('calibrate','check'):
        mode=commands.add_parser(kind); mode.add_argument('output',type=Path)
        mode.add_argument('--e0-engine',required=True); mode.add_argument('--e0-deployment',required=True)
        mode.add_argument('--reference',nargs=2,required=True,metavar=('ARCHIVE','PIN'))
        for arm in (('e0a','positive') if kind=='calibrate' else ('candidate','calibration')):
            mode.add_argument('--'+arm,nargs=2,required=True,metavar=('PATH','PIN'))
    a=p.parse_args(); q.need(not a.output.exists(),'audit receipt is write-once')
    def take(x): return evidence(Path(x[0]),x[1])
    ref=take(a.reference); opts=dict(e0_engine=a.e0_engine,e0_deployment=a.e0_deployment)
    common=dict(schema=SCHEMA,scoring=False,holdout=False,paid_calls=0,driver=file_sha256(Path(__file__)),
                e0_engine=a.e0_engine,e0_deployment=a.e0_deployment,reference=a.reference[1])
    if a.mode=='calibrate':
        result=calibrate(take(a.e0a),ref,take(a.positive),**opts)
        row=dict(common,role='calibration',e0a=a.e0a[1],positive=a.positive[1],**result)
    else:
        cb=read_pin(Path(a.calibration[0]),a.calibration[1],cap=128<<10)
        q.need(all(cb[k]==v for k,v in common.items()) and cb['role']=='calibration','audit reference or calibration differs')
        result=compare(take(a.candidate),ref,**opts)
        row=dict(common,role='audit',calibration=a.calibration[1],candidate=a.candidate[1],comparison=result,
                 **{k:v for k,v in hints(result,cb['reference_envelope']).items() if k!='scoring'})
    print(write(a.output,row))


if __name__=='__main__':
    main()
