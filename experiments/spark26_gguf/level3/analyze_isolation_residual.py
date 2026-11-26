"""Decompose an archived DEV trajectory verdict without modifying it or retuning bounds."""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256


def decompose(candidate, teacher, limits, *, separate_draw=False):
    report = q.compare(candidate, teacher)
    verdict = q.verdict(report, limits)
    plan = dict(zip((job['key'] for job in q.teacher_plan(candidate)), teacher['rows'], strict=True))
    jobs = {job['key']:job for job in q.teacher_plan(candidate)}
    records = []
    for pair in candidate['isolation']:
        trigger = pair['trigger']
        for i in q.TAIL:
            row = dict(trigger=trigger, step=i)
            residuals, draw_tokens = {}, {}
            for arm in ('complete','cancel'):
                survivor = pair[arm][0]
                key = f'i/{trigger}/{arm}/{i}'
                ref = dict(zip(jobs[key]['wanted'],plan[key]['values'],strict=True))
                tokens = (*q.MARKERS,survivor['ids'][i]); emitted = survivor['probs'][i]
                residuals[arm] = [math.exp(p)-math.exp(ref[t]) for t,p in zip(tokens,emitted,strict=True)]
                row[arm] = dict(eos_log_delta=emitted[q.MARKERS.index(q.EOS)]-ref[q.EOS],
                    max_abs_probability_delta=max(map(abs,residuals[arm])))
                if separate_draw:
                    marker_peak = max(map(abs, residuals[arm][:-1]))
                    draw_delta = abs(residuals[arm][-1])
                    draw_tokens[arm] = survivor['ids'][i]
                    row[arm].update(marker_probability_delta_max=marker_peak,
                                    drawn_probability_delta_abs=draw_delta,
                                    drawn_probability_strictly_dominates=draw_delta > marker_peak)
            left, right = jobs[f'i/{trigger}/complete/{i}'], jobs[f'i/{trigger}/cancel/{i}']
            row['same_survivor_prefix'] = left['ids']==right['ids']
            if separate_draw:
                row['same_drawn_token'] = draw_tokens['complete'] == draw_tokens['cancel']
                row['paired_drawn_probability_residual_abs'] = (
                    abs(residuals['complete'][-1]-residuals['cancel'][-1]) if row['same_drawn_token'] else None)
            row['paired_marker_residual_max'] = max(abs(a-b) for a,b in zip(
                    residuals['complete'][:-1],residuals['cancel'][:-1],strict=True))
            left, right = (row[arm]['eos_log_delta'] for arm in ('complete','cancel'))
            row['paired_eos_log_residual_abs'] = abs(left-right)
            row['common_eos_log_residual'] = (left+right)/2
            records.append(row)
    return dict(verdict=verdict,limits=limits,points=records)


def pinned_json(path, pin, max_bytes=24<<20):
    q.need(path.is_file() and not path.is_symlink() and path.stat().st_size<=max_bytes
           and file_sha256(path)==pin,'DEV evidence fingerprint changed')
    return json.loads(path.read_bytes())


def run(job, candidate_pin, teacher_pin, calibration, calibration_pin, output, *, separate_draw=False):
    cal = pinned_json(calibration,calibration_pin)
    q.need(cal['limits']==q.envelope([arm['metrics'] for arm in cal['arms']]),'incomplete calibration')
    witnesses, rows = [], []
    for role,pin in (('candidate',candidate_pin),('teacher',teacher_pin)):
        folder = job/('trajectory-'+role)
        proof = pinned_json(folder/'trajectory.json',pin)
        q.need(proof['role']==role and proof['protocol']==cal['protocol']==q.VERSION and proof['code']==cal['code'],
               'trajectory role or code differs from calibration')
        rows.append(pinned_json(folder/'observer.stdout',proof['files']['observer.stdout']))
        witnesses.append(proof)
    c,t = witnesses
    q.need(t['deployment'] in cal['deployments'].values() and c['binding']['service_id']!=t['binding']['service_id'] and
           c['binding']['end_ns']<=t['binding']['start_ns'] and c['weights']==t['weights'], 'not an independent DEV teacher')
    return write(output,dict(scope='analysis_of_frozen_dev_data_never_a_new_verdict',engine=c['binding']['engine'],
                original_pins=dict(candidate=candidate_pin,teacher=teacher_pin,calibration=calibration_pin),
                result=decompose(*rows,cal['limits'],separate_draw=separate_draw),holdout=False,paid_calls=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job',type=Path); ap.add_argument('candidate_pin'); ap.add_argument('teacher_pin')
    ap.add_argument('calibration',type=Path); ap.add_argument('calibration_pin'); ap.add_argument('new_output',type=Path)
    ap.add_argument('--separate-draw', action='store_true',
                    help='also distinguish the fixed markers from the drawn-token residual')
    a=ap.parse_args(); print(run(a.job,a.candidate_pin,a.teacher_pin,a.calibration,a.calibration_pin,a.new_output,
                                 separate_draw=a.separate_draw))
