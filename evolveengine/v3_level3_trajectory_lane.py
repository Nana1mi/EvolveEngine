"""Host-only binding of two isolated deployments to a development trajectory observation."""
import json
import zlib
from pathlib import Path

from .v3_harness import canonical, digest
from .v3_level3_collect import DockerObserver, identity, need, source_bytes, stage_inputs, write
from .v3_level3_evidence import _external_identity, file_sha256
from . import v3_level3_trajectory as contract


def candidate_jobs(raw):
    """Only the candidate payload is recoverable; never soften host pins/identity or E0 proofs."""
    from .v3_level3_worker import ObservationUnavailable
    try:
        jobs = contract.teacher_plan(raw)
        # Do not spend a teacher service on unreadable or non-normalized distributions.
        for row in raw['decode']:
            for vector in row['vectors']:
                contract.unpack(vector)
        return jobs
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, zlib.error) as exc:
        raise ObservationUnavailable('incomplete candidate trajectory payload') from exc


SPLIT_PROTOCOL = 'split-absolute-quality-causal-cancel-dev-v1'
SPLIT_CASES = ('autoregressive_full_vocab','mixed_tail_absolute','paired_cancel_numeric')


def trajectory_cases(record, protocol):
    """Derive the archive case outcomes from the replayable host verdict."""
    need(record['protocol']==protocol,'trajectory verdict protocol differs')
    if protocol==contract.VERSION:
        return {case:'fail' if any(k.startswith(prefix) for k in record['failed']) else 'pass'
                for case,prefix in zip(('paired_cancel_numeric','autoregressive_full_vocab'),('iso_','decode_'))}
    need(protocol==SPLIT_PROTOCOL and record['live_gate'] is True and set(record['groups'])==set(SPLIT_CASES),
         'missing live split quality groups')
    outcomes={}
    for case in SPLIT_CASES:
        group=record['groups'][case]
        status='violation' if group['violations'] else 'insufficient_evidence' if group['insufficient'] else 'valid'
        need(group['status']==status,'inconsistent split quality group status')
        outcomes[case]={'valid':'pass','violation':'fail','insufficient_evidence':'insufficient_evidence'}[status]
    overall='violation' if 'fail' in outcomes.values() else 'insufficient_evidence' if 'insufficient_evidence' in outcomes.values() else 'valid'
    need(record['status']==overall,'inconsistent split quality overall status')
    return outcomes


class TrajectoryCollector:
    protocol = contract.VERSION
    def __init__(self, quality, scripts):
        self.quality = quality
        self.files = {n:source_bytes(p) for n,p in scripts.items()}
        need(set(self.files)=={'observe_trajectory.py','trajectory_contract.py','observe_timing.py','timing.py',
                                'observe_text_timing.py'}, 'incomplete trajectory observer code')
        self.code = digest(canonical({k:digest(v) for k,v in self.files.items()}))
        self.observer = DockerObserver(quality.host,quality.service_spec)

    def collect(self, service, root, deadline, *, jobs=None):
        root.mkdir(mode=0o700)
        plan=dict(protocol=contract.VERSION,role='candidate' if jobs is None else 'teacher')
        if jobs is not None: plan['jobs']=jobs
        stage_inputs(root/'inputs',{**self.files,'trajectory-plan.json':canonical(plan)})
        self.observer(service,root/'inputs',root,deadline,script='observe_trajectory.py',output_cap=24<<20)
        q=self.quality
        p,binding=identity(q.host,service,q.service_spec,root,'develop_quality',q.policy,deadline,q.variant)
        need(not _external_identity(binding,p,root/'external-identity.json'),'trajectory runtime identity mismatch')
        files=['observer.stdout','observer.stderr','observer.inspect.json','external-identity.json','service.inspect.json',
               'host-mappings.json','inputs/trajectory-plan.json',*('inputs/'+name for name in self.files)]
        return write(root/'trajectory.json',dict(binding=binding,role=plan['role'],protocol=contract.VERSION,code=self.code,
             weights=dict(p.weights),loaded=dict(p.loaded_artifacts),deployment=service.deployment.identity,
             files={name:file_sha256(root/name) for name in files}))

    def read(self, root, pin, deployment, *, jobs=None):
        from dataclasses import replace
        need(file_sha256(root/'trajectory.json')==pin,'trajectory receipt changed')
        proof=json.loads((root/'trajectory.json').read_bytes()); bind=proof['binding']
        role='candidate' if jobs is None else 'teacher'
        p=self.quality.policy
        need((proof['role'],proof['protocol'],proof['code'],proof['deployment'])==(role,contract.VERSION,self.code,deployment.identity)
             and (bind['engine'],bind['boot_id'],bind['device_uuid'],bind['worker_id'],bind['origin'])==
             (deployment.candidate,p.boot_id,p.device_uuid,p.worker_id,'hardware_collection') and
             proof['weights']==dict(p.weights) and proof['loaded']==deployment.native_mapping(self.quality.variant)[1],
             'trajectory not bound to registered deployment')
        required={'observer.stdout','observer.stderr','observer.inspect.json','external-identity.json','service.inspect.json',
                  'host-mappings.json','inputs/trajectory-plan.json',*('inputs/'+name for name in self.files)}
        need(required==proof['files'].keys() and all(file_sha256(root/k)==v for k,v in proof['files'].items()) and
             all(proof['files']['inputs/'+k]==digest(v) for k,v in self.files.items()),'trajectory evidence changed')
        plan=dict(protocol=contract.VERSION,role=role)
        if jobs is not None: plan['jobs']=jobs
        need((root/'inputs/trajectory-plan.json').read_bytes()==canonical(plan),'teacher forcing plan differs')
        need(not _external_identity(bind,replace(p,engine=deployment.candidate,loaded_artifacts=proof['loaded']),
                                    root/'external-identity.json'),'trajectory weight or mapping violation')
        data=json.loads((root/'observer.stdout').read_bytes()); need(data['role']==role,'observer response belongs to another role')
        return data, bind

    def report(self, root, pins, deployment, e0):
        return contract.compare(*self.independent_observations(root,pins,deployment,e0))

    def independent_observations(self, root, pins, deployment, e0):
        c, cb=self.read(root/'trajectory-candidate',pins[0],deployment)
        t, tb=self.read(root/'trajectory-teacher',pins[1],e0,jobs=contract.teacher_plan(c))
        need(cb['service_id']!=tb['service_id'] and cb['run_id']!=tb['run_id'] and cb['end_ns']<=tb['start_ns'],
             'expected sequential isolated candidate and E0 services')
        return c,t

    def activate(self, ref, registration, e0):
        path=Path(ref['path']); need(path.is_absolute() and path.resolve(strict=True)==path and
            file_sha256(path)==ref['sha256'],'unverified trajectory calibration')
        raw=json.loads(path.read_bytes()); arms=raw['arms']
        need(raw['protocol']==contract.VERSION and raw['registration_sha256']==registration and raw['e0']==e0 and
             raw['code']==self.code and [(r['candidate'],r['teacher']) for r in arms]==[('e0a','e0b'),('e0b','e0a')]
             and set(raw['deployments'])=={'e0a','e0b'} and len(set(raw['deployments'].values()))==2 and
             all(len({r[k] for r in arms})==2 for k in ('candidate_receipt','teacher_receipt')) and
             file_sha256(path.parent/'frozen-plan.json')==raw['plan_sha256'] and
             raw['limits']==contract.envelope([r['metrics'] for r in arms]),'unbound trajectory calibration')
        self.calibration_pin,self.limits,self.teacher_deployments=ref['sha256'],raw['limits'],set(raw['deployments'].values())

    def screened(self, root, pins, deployment, e0):
        need(e0.identity in self.teacher_deployments,'unregistered E0 teacher deployment')
        report=self.report(root,pins,deployment,e0)
        info=json.loads((root/'trajectory-candidate'/'trajectory.json').read_bytes())
        return dict(contract.verdict(report,self.limits),engine=deployment.candidate,service=info['binding']['service_id'],
                    calibration=self.calibration_pin,candidate_receipt=pins[0],teacher_receipt=pins[1],teacher_deployment=e0.identity)


def scripts(here):
    return {'observe_trajectory.py':here/'observe_trajectory.py','trajectory_contract.py':Path(contract.__file__),
            'observe_timing.py':here/'observe_timing.py','observe_text_timing.py':here/'observe_text_timing.py',
            'timing.py':here.parent/'v3_dev_rpc'/'timing.py'}
