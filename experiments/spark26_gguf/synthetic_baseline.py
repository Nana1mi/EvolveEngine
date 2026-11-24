"""Frozen warmed E0 calibration and separate engine-profile requests."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

BASE = 'http://127.0.0.1:30091'
WORKLOAD = Path(__file__).with_name('calibration_workload.json')


def emit(**row):
    print(json.dumps(row, sort_keys=True), flush=True)


def mark(marker):
    print(f'EE_CALIBRATION {marker}', file=sys.stderr, flush=True)


def open_json(route, payload):
    headers = {'Content-Type': 'application/json'}
    if route == '/start_profile':
        headers['Authorization'] = 'Bearer ' + os.environ['SGLANG_ADMIN_API_KEY']
    req = Request(BASE + route, json.dumps(payload).encode(), headers)
    return build_opener(ProxyHandler({})).open(req, timeout=60)


def wave(prefix, spec):
    n, width, output = (spec[k] for k in ('input_tokens', 'width', 'output_tokens'))
    rids = [f'{prefix}-{i}' for i in range(width)]
    inputs = [[6510+37*lane+(i % 37) for i in range(n)] for lane in range(width)]
    payload = dict(rid=rids[0] if width == 1 else rids, input_ids=inputs[0] if width == 1 else inputs,
                   stream=True, sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=output))
    first, final, done = {}, {}, False
    start = time.perf_counter()
    with open_json('/generate', payload) as response:
        if response.status != 200 or response.headers.get_content_type() != 'text/event-stream':
            raise ValueError('Expected HTTP 200 native SSE')
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            msg = json.loads(data)
            if 'error' in msg:
                raise RuntimeError(msg['error'])
            meta = msg['meta_info']
            rid = meta['id']
            if rid not in rids:
                raise ValueError(f'Unexpected id {rid}')
            if meta['completion_tokens'] > 0 and rid not in first:
                first[rid] = (time.perf_counter()-start)*1000
            if meta['finish_reason'] is not None:
                if rid in final or (meta['prompt_tokens'], meta['completion_tokens'], meta['finish_reason']['type']) != (n, output, 'length'):
                    raise ValueError(f'Unexpected completion {meta}')
                ids = msg['output_ids']
                if len(ids) != output or not all(type(i) is int for i in ids):
                    raise ValueError('Invalid cumulative token IDs')
                final[rid] = {'output_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                              'input_tokens': n, 'output_tokens': output,
                              'finish': 'length'}
    elapsed = time.perf_counter()-start
    if not done or set(first) != set(rids) or set(final) != set(rids):
        raise ValueError(f'Missing outputs or [DONE] for {prefix}')
    return dict(wall_seconds=elapsed, requests=[dict(final[rid], rid=rid, first_token_ms=first[rid]) for rid in rids])


def calibrate(lock, label):
    pair = next(p for p in lock['pairs'] if label in p['labels'])
    emit(kind='run_identity', label=label, workload_sha256=hashlib.sha256(WORKLOAD.read_bytes()).hexdigest())
    for name in pair['shape_order']:
        spec = lock['shapes'][name]
        for phase, count in [('warmup', lock['warmup_waves']), ('timed', lock['timed_waves'])]:
            for i in range(count):
                mark(f'{name} {phase} wave-{i}')
                row = wave(f'{label}-{name}-{phase}-{i}', spec)
                emit(kind='wave', shape=name, phase=phase, index=i, **row)
    mark('complete')


def profile(lock):
    # Different process and GPU timing only; never put these requests in E0 noise results.
    for name in ('short1', 'long2'):
        spec = lock['shapes'][name]
        for i in range(lock['warmup_waves']):
            wave(f'{name}-prof-warm-{i}', spec)
        mark(f'{name} profile')
        with open_json('/start_profile', dict(output_dir='/profile-out', profile_prefix=name,
                       activities=['CPU','GPU'], profile_by_stage=True, num_steps=2,
                       with_stack=False, record_shapes=os.environ.get("EE_SPARK26_PROFILE_SHAPES") == "1")) as response:
            if response.status != 200:
                raise RuntimeError(f'/start_profile rejected {name}')
            result = response.read(1024).decode('utf-8', errors='replace').strip()
        emit(kind='profile_start', shape=name, response=result)
        for i in range(4):
            wave(f'{name}-prof-{i}', spec)
        files = sorted(Path('/profile-out').glob(f'{name}*.trace.json.gz'))
        for path in files:
            with gzip.open(path, 'rb') as f:
                if not f.read(1):
                    raise ValueError(f'Empty trace: {path.name}')
        if len(files) < 2:
            raise ValueError(f'Expected prefill and decode traces for {name}: {files}')
        emit(kind='profile_files', shape=name, files=[p.name for p in files])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: synthetic_baseline.py <trial-label|profile>')
    lock = json.loads(WORKLOAD.read_bytes())
    if sys.argv[1] == 'profile':
        profile(lock)
    else:
        calibrate(lock, sys.argv[1])
