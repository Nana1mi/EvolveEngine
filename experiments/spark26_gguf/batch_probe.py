"""Measure the first and subsequent verified joint prefill on diagnostic E0."""
import json
import os
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ENDPOINT = 'http://127.0.0.1:30091/generate'
LANES = ([6510 + i % 37 for i in range(130)], [6547 + i % 37 for i in range(130)])
TOKENS = 16
PHASES = ('first_token_latency', 'prefill_latency', 'before_queue', 'after_prefill',
          'other_queue', 'storage_prefetch_wait', 'e2e_latency')


def emit(**row):
    print(json.dumps(row, sort_keys=True), flush=True)


def cache_metadata():
    root = Path(os.environ['TRITON_CACHE_DIR'])
    files = set()
    for path in root.rglob('*.json'):
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def measure(stage, lanes):
    rids = [f'{stage}-{lane}' for lane in lanes]
    inputs = [LANES[lane] for lane in lanes]
    if len(lanes) == 1:
        rids_input, inputs = rids[0], inputs[0]
    else:
        rids_input = rids
    payload = dict(rid=rids_input, input_ids=inputs, stream=True,
                   sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=TOKENS))
    req = Request(ENDPOINT, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    opener = build_opener(ProxyHandler({}))
    first, final, done = {}, {}, False
    before = cache_metadata()
    start = time.perf_counter()
    with opener.open(req, timeout=60) as response:
        if response.status != 200 or response.headers.get_content_type() != 'text/event-stream':
            raise ValueError(f'Expected HTTP 200 SSE for {stage}')
        for line in response:
            if not line.startswith(b'data:'):
                continue
            event = line[5:].strip()
            if event == b'[DONE]':
                done = True
                break
            data = json.loads(event)
            if 'error' in data:
                raise RuntimeError(data['error'])
            meta = data['meta_info']
            rid = meta['id']
            if rid not in rids:
                raise ValueError(f'Unexpected request id: {rid}')
            if meta['completion_tokens'] > 0 and rid not in first:
                first[rid] = (time.perf_counter()-start)*1000
            if meta['finish_reason'] is not None:
                if rid in final:
                    raise ValueError(f'Duplicate completion: {rid}')
                final[rid] = (meta, data['output_ids'])
    if not done or set(first) != set(rids) or set(final) != set(rids):
        raise ValueError(f'Missing stream terminator or outputs for {stage}')
    after = cache_metadata()
    for rid in rids:
        meta, ids = final[rid]
        if (meta['prompt_tokens'], meta['completion_tokens'], meta['finish_reason']['type']) != (130, TOKENS, 'length'):
            raise ValueError(f'Unexpected termination or token count: {meta!r}')
        if len(ids) != TOKENS:
            raise ValueError(f'Expected {TOKENS} cumulative output IDs, got {len(ids)}')
        if not all(k in meta for k in ('first_token_latency', 'prefill_latency')):
            raise ValueError(f'Missing server timing breakdown: {meta!r}')
        emit(kind='request', stage=stage, rid=rid, width=len(lanes), client_ttft_ms=first[rid],
             **{f'{k}_ms': meta[k]*1000 for k in PHASES if isinstance(meta.get(k), (int, float))},
             prompt_tokens=meta['prompt_tokens'], completion_tokens=meta['completion_tokens'],
             token_ids=ids)
    emit(kind='cache', stage=stage, before=len(before), after=len(after), new=sorted(after-before))
    return [final[rid][1] for rid in rids]


def run():
    pre = [measure('single-a', (0,))[0], measure('single-b', (1,))[0]]
    pairs = [measure(f'paired-{i}', (0, 1)) for i in range(1, 7)]
    post = [measure('post-a', (0,))[0], measure('post-b', (1,))[0]]
    emit(kind='token_ids_check', single_before_after=pre == post,
         same_across_pairs=all(row == pairs[0] for row in pairs), single_vs_pair=pre == pairs[0])


if __name__ == '__main__':
    run()
