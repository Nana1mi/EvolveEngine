"""Independent Docker client; fixed dev inputs, no candidate file access."""
import concurrent.futures as cf
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import build_opener,ProxyHandler

import numeric
import timing


def await_ready():
    url='http://127.0.0.1:30091/health'
    client=build_opener(ProxyHandler({}));end=time.monotonic()+180
    while time.monotonic()<end:
        try:
            with client.open(url,timeout=2) as r:
                if r.status==200:return
        except OSError:pass
        time.sleep(.4)
    raise RuntimeError('service health not observed')


def residency_stage(stage):
    phase = Path('/ee-phase')
    marker = phase / (stage + '.new')
    marker.write_text(stage, encoding='ascii')
    marker.replace(phase / stage)
    reply = timing.request(0, 32, 4, 'ee-residency-'+stage, threading.Barrier(1))
    numeric.emit('residency_stage', stage=stage, request=reply['rid'], output_sha256=reply['sha256'])


def quality():
    residency_stage('ready')
    # The numeric request marker is local to the trusted client, not the candidate.
    for i in range(32):numeric.emit('numeric',ok=True,**numeric.numeric('worker-dev',i))
    for stage in (0,8,32):numeric.emit('kv',stage=stage,ok=True,**numeric.mixed('worker-dev',stage))
    subprocess.run([sys.executable,'/dev-probe/legacy_eos.py','worker-dev'],check=True)
    residency_stage('post_requests')


def measure():
    plan=json.loads(Path('/dev-probe/plan.json').read_bytes());
    for scene,shape in plan['scenes'].items():
        for idx in range(plan['warm_waves']+plan['timed_waves']):
            width=len(shape);g=threading.Barrier(width)
            with cf.ThreadPoolExecutor(max_workers=width) as ex:
                futures=[ex.submit(timing.request,i,n,k,f'dev-rpc-{scene}-{idx}-{i}',g) for i,(n,k) in enumerate(shape)]; rows=[f.result() for f in futures]
            dt=(max(r['done_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
            overlap=(min(r['last_ns'] for r in rows)-max(r['first_ns'] for r in rows))/1e6
            assert width==1 or overlap>0,'four-lane client intervals did not overlap'
            value=(rows[0]['decode_tps'] if scene in ('W1','W2') else 1000/rows[0]['ttft_ms'] if scene=='W3' else sum(r['output'] for r in rows)/dt)
            numeric.emit('wave',scene=scene,index=idx,timed=idx>=plan['warm_waves'],width=width,P=value,overlap_ms=overlap,wall_s=dt,requests=rows)

if __name__=='__main__':
    await_ready()
    {'develop_quality':quality,'develop_measure':measure}[sys.argv[1]]()
