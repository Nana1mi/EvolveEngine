"""Host-only sampling draws. One billable node operation per arm; no timing or Pi inputs."""
import json
from pathlib import Path
from secrets import token_hex
from time import monotonic

from .v3_harness import canonical, digest
from . import v3_level3_sampling as stats
from .v3_level3_collect import identity, need, stage_inputs, write
from .v3_level3_evidence import _external_identity, file_sha256
from .v3_level3_run import WorkerResult
from .v3_level3_worker import ObservationUnavailable


class SamplingLane:
    def __init__(self, worker, quality):
        self.worker,self.quality=worker,quality
        self.draws,self.records={},{}

    def samples(self, root):
        rows=[json.loads(x) for x in (root/'observer.stdout').read_bytes().splitlines()]
        need([r['case'] for r in rows]==list(stats.CASES) and all(r['kind']=='sampling' for r in rows),'missing independent draws')
        return stats.checked_samples({str(r['case']):r['sequences'] for r in rows})

    def run(self, engine, build, cap, spec):
        mark,stage,role=stats.challenge(spec); w=self.worker;q=self.quality
        need(not w.quarantined and build.status=='built' and build.engine==engine.digest and w.verify(build), 'sampling requires verified build')
        key=(engine.digest,build.evidence,mark,stage,role); base=key[:3]
        need(key not in self.draws and (stage==1 or self.draws[(*base,1,'trial')].data['decision']=='inconclusive'), 'unregistered or repeated sampling look')
        ref=None if role=='reference' else self.draws[(*base,stage,'reference')]
        if ref is not None: need(ref.status=='sampled' and self.verify(ref),'unverified E0 sampling reference')
        dep=w.trajectory_e0 if ref is None else w.deployments[build.evidence]
        need(dep is not None,'no calibrated E0 deployment'); self.draws[key]=None  # claim before I/O
        started=monotonic();deadline=started+cap/1000; root=w.root/('sample-'+token_hex(12));root.mkdir(mode=0o700)
        write(root/'reserved.json',dict(engine=engine.digest,build=build.evidence,protocol=stats.VERSION,challenge=spec,cap_ms=cap))
        stage_inputs(root/'inputs', {'observe_quality.py':q.files['observe_quality.py'],'numeric.py':q.files['numeric.py'],
                    'sampling-only.json':canonical(dict(cases=stats.CASES,samples=stats.SAMPLES))})
        data=dict(protocol=stats.VERSION,challenge=dict(spec),deployment=dep.identity,observed_engine=dep.candidate)
        try:
            with w.exclusive(deadline),w.service.start(dep,'ee-l3-sample-'+token_hex(8),deadline) as live:
                q.observer(live,root/'inputs',root,deadline,script='observe_quality.py',output_cap=4<<20)
                policy,binding=identity(q.host,live,q.service_spec,root,'develop_quality',q.policy,deadline,q.variant)
                need(not _external_identity(binding,policy,root/'external-identity.json'),'sampling identity mismatch')
            try: draws=self.samples(root)
            except (KeyError,ValueError,TypeError) as exc: raise ObservationUnavailable('missing sampled sequence evidence') from exc
            if ref is None: data.update(decision='reference')
            else:
                refroot=self.records[ref.evidence][1]; test=stats.compare(self.samples(refroot),draws,alpha=stats.ALPHA)
                need(binding['service_id']!=ref.data['service'] and binding['start_ns']>=ref.data['end_ns'],'sampling arms overlap')
                data.update(decision=stats.decision(test),reference=ref.evidence,test=test)
            data.update(service=binding['service_id'],end_ns=binding['end_ns']);status='sampled'
        except ObservationUnavailable:
            status='insufficient_evidence';data.update(decision='incomplete')
        except BaseException:
            w.quarantined=True;raise
        files={p.relative_to(root).as_posix():file_sha256(p) for p in root.rglob('*') if p.is_file()}
        elapsed=round((monotonic()-started)*1000); need(elapsed<=cap,'sampling lease overrun')
        data['files_sha256']=digest(canonical(files));fields=dict(engine=engine.digest,build=build.evidence,operation='develop_quality',
            node_ms=elapsed,status=status,data=data);pin=write(root/'result.json',dict(result=fields,files=files))
        result=WorkerResult(**fields,evidence=pin);self.records[pin]=(result,root,files);self.draws[key]=result;return result

    def verify(self, result):
        saved,root,files=self.records[result.evidence]
        if saved!=result or file_sha256(root/'result.json')!=result.evidence or any(file_sha256(root/k)!=v for k,v in files.items()): return False
        if result.status=='sampled' and result.data['challenge']['role']=='trial':
            ref=self.records[result.data['reference']];test=stats.compare(self.samples(ref[1]),self.samples(root),alpha=stats.ALPHA)
            return self.verify(ref[0]) and test==result.data['test'] and stats.decision(test)==result.data['decision']
        return result.data['decision'] in ('incomplete','reference')
