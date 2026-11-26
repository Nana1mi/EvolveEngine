"""Registered live binding for the hardware-checked split development quality rule.

The baseline registration and its historical receipts remain unchanged. An
explicit, separately pinned overlay permits only the host adapter changes
needed to attach the validated observer and to carry three-state evidence.
"""
import json
from pathlib import Path

from experiments.spark26_gguf.level3 import late_trajectory as late, split_quality as split
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import need
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import SPLIT_PROTOCOL, TrajectoryCollector, scripts, trajectory_cases
from evolveengine.v3_level3_worker import ObservationUnavailable

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCHEMA='spark26-live-split-quality-registration-v1'
OVERLAY_CHANGES=frozenset((*('experiments/spark26_gguf/level3/'+n for n in ('node_preflight.py','worker_stdio.py')),
    *('evolveengine/v3_level3_'+n+'.py' for n in ('collect','evidence','trajectory_lane','worker'))))
OPTIONAL_OVERLAY_CHANGES = frozenset({'experiments/spark26_gguf/level3/profile_control.py'})
TEACHER_SOURCE={
    'registration': {'path':'/var/tmp/ee-l3-thinking-chat.X45fJZlz/build-registration.json',
                     'sha256':'2b2dc1aabb65e2a01c2ccbdc0b97aadadbe3a4167c11589af64f3f7fb154bb34'},
    'build': {'path':'/var/tmp/ee-l3-e0e0.v5d/e0b-develop_build.json',
              'sha256':'1b8cc3e190c0717f64e79bb8a1aa9f10e7e598c062730726c41fc794171e9c66'},
}
PROOFS={
    'e0-plan': 'caf4aa2e68b8df7254a693975f27158c91e6eb49abe8d8224ae2a6e23e6d68c9',
    'e0-result': '70c5542f31624a7141714d282961115ffa253dcfac216df0b233adbfe7e0125a',
    'fault-replay': 'c4df178bdd3c1d74b544e03802d2f08046c8bcb147f05b27a95e19e3aa43a652',
    'compound-plan': '6231f84ee04e07ec83700d7c62b19446fb00e6b550d75b79f4bbdeb844485234',
    'compound-result': 'd57ac76bd997ede630dfde7feac5d7308a19f31db769d7af962bce633e08f34e',
}


def pin(x):
    return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def read(path, sha, cap=128<<10):
    need(pin(sha) and path.resolve(strict=True)==path and path.is_file() and path.stat().st_size<=cap,
         'invalid split evidence path or fingerprint')
    raw=path.read_bytes(); row=json.loads(raw)
    need(digest(raw)==sha and raw==canonical(row)+b'\n','split evidence bytes changed')
    return row


def extra_sources():
    return {p.name:file_sha256(p) for p in (Path(__file__),Path(late.__file__),Path(late.cold.__file__),
            Path(late.partial_tv.__file__),Path(split.__file__),HERE/'isolation_cache_parity.py')}


def verify_overlay(baseline, live, registration):
    changes=registration['implementation_overlay']
    changed_paths = set(changes)
    need(baseline.keys()==live.keys() and OVERLAY_CHANGES <= changed_paths <= OVERLAY_CHANGES | OPTIONAL_OVERLAY_CHANGES
         and changed_paths <= baseline.keys() and {k for k in baseline if baseline[k]!=live[k]} == changed_paths
         and all(changes[k]==[baseline[k],live[k]] for k in changes),
         'live split overlay changes unregistered baseline code')


def read_registration(path, sha, baseline, calibration):
    row=read(path,sha)
    need(set(row)=={'schema','protocol','baseline','calibration','implementation_overlay','sources','proofs','observer','teacher','teacher_source'}
         and (row['schema'],row['protocol'],row['baseline'],row['calibration'])==(SCHEMA,SPLIT_PROTOCOL,baseline,calibration)
         and row['sources']==extra_sources() and row['teacher_source']==TEACHER_SOURCE and row['proofs']==PROOFS
         and pin(row['observer']) and pin(row['teacher']),
         'live split registration differs from the signed observation protocol')
    proof={k:read(path.parent/(k+'.json'),v) for k,v in PROOFS.items()}
    ep,er,fr,cp,cr=(proof[k] for k in PROOFS)
    need(er['plan']==PROOFS['e0-plan'] and cr['plan']==PROOFS['compound-plan'] and fr['source']==PROOFS['e0-result'] and
         cp['e0_late']==PROOFS['e0-result'] and cp['e0_controls']==PROOFS['fault-replay'] and
         ep['teacher']==cp['teacher']==row['teacher'] and ep['observer']==cp['observer']==row['observer'] and
         ep['registration']==baseline and cp['calibration_v2']==fr['trajectory_v2']==calibration and
         fr['metric_rule']==cp['metric_rule']==row['sources']['split_quality.py'] and
         ep['helper']==row['sources']['isolation_cache_parity.py'] and ep['bounds']==row['sources']['partial_tv.py'] and
         fr['checks']['baseline']['status']=='valid' and cr['assessment']['status']=='valid',
         'wrong split shadow control or teacher evidence')
    for k,case in (('common_eos','mixed_tail_absolute'),('paired_eos','paired_cancel_numeric')):
        bad=fr['checks']['synthetic_controls'][k]['groups']
        need(set(bad)==set(split.GROUPS) and all(v['status']==('violation' if n==case else 'valid') for n,v in bad.items()),
             'fault replay did not isolate the registered group')
    return row



def restore_teacher(root, settings, contract, e0, backend, registered):
    from experiments.spark26_gguf.level3 import replay_e0
    from evolveengine.v3_level3_deploy import stage_deployment
    al=registered['teacher_source']; reg=al['registration']; build=al['build']
    cfg=read(Path(reg['path']),reg['sha256']); replay_e0.check_build_compatibility(settings,cfg)
    native,result=replay_e0.restore_build(Path(build['path']),build['sha256'],contract=contract,e0=e0,backend=backend,
                                          registration=cfg,registration_pin=reg['sha256'],arm='e0b')
    teacher=stage_deployment(native,root/'split-e0-teacher')
    need(teacher.candidate==e0.digest and teacher.identity==result['data']['deployment']==registered['teacher'],
         'split teacher does not match the isolated E0 artifact')
    return teacher


class LiveTailLane(late.TailLane):
    protocol=SPLIT_PROTOCOL

    def __init__(self, quality, paths, helper):
        super().__init__(quality,paths,helper)
        self.paths=paths

    def activate(self, ref, registration, e0, *, split_ref, registered):
        legacy=TrajectoryCollector(self.quality,self.paths)
        legacy.activate(ref,registration,e0)
        need(self.parent_code==legacy.code and self.code==registered['observer'] and
             registered['teacher'] in legacy.teacher_deployments,'observer bytes or independent E0 teacher differ')
        self.calibration_pin,self.limits=legacy.calibration_pin,legacy.limits
        self.teacher_deployments={registered['teacher']}
        self.split_pin=split_ref['sha256']

    def read(self, root, pin, deployment, *, jobs=None):
        # Only malformed observation data is recoverable. A changed archive,
        # registered deployment or teacher plan must not be treated as a retry.
        data,binding=TrajectoryCollector.read(self,root,pin,deployment,jobs=jobs)
        if jobs is None:
            try:
                late.cold.verify_cache(data)
                for pair in data['isolation']:
                    for arm in ('complete','cancel'):
                        for step in late.q.TAIL: late.view(pair[arm][0],step)
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                raise ObservationUnavailable('incomplete cold-pair late trajectory') from exc
        return data,binding

    def report(self, root, pins, deployment, e0):
        return late.tail_report(*self.independent_observations(root,pins,deployment,e0),require_e0_ceiling=False)

    def screened(self, root, pins, deployment, e0):
        need(e0.identity in self.teacher_deployments,'unregistered split E0 teacher deployment')
        report=self.report(root,pins,deployment,e0)
        cproof=read(root/'trajectory-candidate'/'trajectory.json',pins[0])
        assessment=split.assess(report,self.limits)
        # Attachment is only emitted after replaying the candidate/teacher pins,
        # cache state, identity and byte-exact plan on this registered lane.
        record=dict(assessment,live_gate=True,engine=deployment.candidate,service=cproof['binding']['service_id'],
                    calibration=self.calibration_pin,split_registration=self.split_pin,observer=self.code,
                    candidate_receipt=pins[0],teacher_receipt=pins[1],teacher_deployment=e0.identity,
                    evidence_summary=digest(canonical(report)))
        trajectory_cases(record,self.protocol)
        return record


def enable(quality, settings, ref, split_ref, registration, e0, registered):
    from evolveengine.v3_level3_eos import grade
    lane=LiveTailLane(quality,scripts(HERE),HERE/'isolation_cache_parity.py')
    lane.activate(ref,registration,e0.digest,split_ref=split_ref,registered=registered)
    quality.enable_trajectory(ref['sha256'],lane.code,
        lambda rows:grade(rows,Path(settings['anchor_d3']),Path(settings['anchor_r1']),paired=True),
        split_pin=split_ref['sha256'])
    return lane
