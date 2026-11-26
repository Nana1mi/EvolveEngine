"""Independent W1–W4 timing client executed only inside the trusted observer."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener

from timing import request


def wait_ready():
    end = monotonic()+180
    client = build_opener(ProxyHandler({}))
    while monotonic() < end:
        try:
            with client.open('http://127.0.0.1:30091/health',timeout=2) as reply:
                if reply.status==200:
                    return
        except OSError:
            sleep(.25)
    raise RuntimeError('no candidate health witness')


def main():
    plan = json.loads(Path('/observer/workload.json').read_bytes())
    wait_ready()
    for scene,spec in plan['scenes'].items():
        for turn in range(plan['warm_waves']+plan['timed_waves']):
            with ThreadPoolExecutor(max_workers=len(spec)) as pool:
                gate = Barrier(len(spec))
                jobs = [pool.submit(request,i,n,out,f'level3-{scene}-{turn}-{i}',gate) for i,(n,out) in enumerate(spec)]
                results = [j.result() for j in jobs]
            rows = [dict(rid=r['rid'],input=r['input'],output=r['output'],sha256=r['sha256'],
                         start_ns=r['start_ns'],first_ns=r['first_ns'],last_ns=r['last_ns'],done_ns=r['done_ns'],
                         first_token_unambiguous=True,stream_complete=True) for r in results]
            print(json.dumps(dict(scene=scene,index=turn,timed=turn>=plan['warm_waves'],requests=rows)),flush=True)


if __name__=='__main__':
    main()
