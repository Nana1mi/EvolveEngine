"""Replay controlled observation faults on *copies* of registered DEV hardware captures.

Not a candidate, not a GPU negative control, and never a way to generate tolerances.
The original capture must first reproduce the signed E0 maxima without modification.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from copy import deepcopy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_collect import write


def controls(candidate, teacher, metrics, limits):
    clean=q.compare(candidate, teacher)
    q.need(clean==metrics and q.verdict(clean, limits)['status']=='valid', 'registered E0 does not replay')
    # Tilt only EOS odds at an actual incremental decode step, without changing the prefix or output length.
    c=deepcopy(candidate); vectors=c['decode'][0]['vectors']; x=q.unpack(vectors[2]); tilt=2.
    shift=math.log1p(math.exp(x[q.EOS])*math.expm1(tilt))
    vectors[2]=q.pack([(min(0.,v+(tilt if i==q.EOS else 0.)-shift),i) for i,v in enumerate(x)])
    decode=q.verdict(q.compare(c,teacher),limits);q.need('decode_eos_logp' in decode['failed'],'decode-only fault escaped')
    # No change in the matched normal-completion arm, prompt, ACK, batch identities, or realized token counts.
    c=deepcopy(candidate); survivor=c['isolation'][0]['cancel'][0]; pos=q.TAIL[-1]; probs=survivor['probs'][pos];
    idx=q.MARKERS.index(q.EOS);probs[idx]=math.log(.01 if math.exp(probs[idx])>.25 else .5)
    if survivor['ids'][pos]==q.EOS:probs[-1]=probs[idx]
    isolate=q.verdict(q.compare(c,teacher),limits);q.need('iso_paired_residual' in isolate['failed'],'post-abort fault escaped')
    return dict(clean='valid',decode_only=decode['failed'],post_cancel_only=isolate['failed'])


def run(root,pin,destination):
    def pinned(path,sha):
        q.need(path.is_file() and not path.is_symlink() and file_sha256(path)==sha,'capture fingerprint differs')
        return json.loads(path.read_bytes())
    spec=pinned(root/'calibration.json',pin);q.need(spec['protocol']==q.VERSION and len(spec['arms'])==2 and
        spec['limits']==q.envelope([a['metrics'] for a in spec['arms']]),'invalid calibration')
    result=[]
    for a in spec['arms']:
        arm=a['candidate'];q.need(arm in ('e0a','e0b'),'unknown hardware arm');rows=[]
        for role in ('candidate','teacher'):
            d=root/arm/('trajectory-'+role);receipt=pinned(d/'trajectory.json',a[role+'_receipt']);q.need(receipt['role']==role,'wrong capture role')
            rows.append(pinned(d/'observer.stdout',receipt['files']['observer.stdout']))
        result.append(dict(arm=arm,**controls(*rows,a['metrics'],spec['limits'])))
    return write(destination,dict(scope='fault_injection_on_copies_of_hardware_observations_not_a_kernel_run',calibration=pin,
             driver=file_sha256(Path(__file__)),arms=result,holdout=False,paid_calls=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('calibration_directory',type=Path)
    cli.add_argument('calibration_sha256');cli.add_argument('new_output',type=Path);args=cli.parse_args()
    print(run(args.calibration_directory,args.calibration_sha256,args.new_output))
