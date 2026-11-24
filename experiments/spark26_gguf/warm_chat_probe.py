"""Recheck application TTFT with the original E0 launcher (no diagnostic flags)."""
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from urllib.request import Request, ProxyHandler, build_opener

# The measurement client is the one baked into the diagnostic E0 image.
sys.path.insert(0, '/opt/spark26')
from e0_probe import call, summarize_block

COLORS = ('红', '蓝', '绿', '白', '橙', '灰', '银', '黑')


def mark(phase):
    print(f'EE_CHAT_PHASE {phase}', file=sys.stderr, flush=True)


def native_prime():
    payload = dict(input_ids=[[6510 + i % 37 for i in range(130)],
                              [6547 + i % 37 for i in range(130)]],
                   rid=['chat-prime-a','chat-prime-b'], stream=False,
                   sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=16))
    request = Request('http://127.0.0.1:30091/generate', json.dumps(payload).encode(),
                      {'Content-Type': 'application/json'})
    t = time.perf_counter()
    with build_opener(ProxyHandler({})).open(request, timeout=60) as response:
        rows = json.load(response)
    ids = set()
    if len(rows) != 2:
        raise ValueError('Prime must return two completions')
    for row in rows:
        m = row['meta_info']
        ids.add(m['id'])
        if (m['prompt_tokens'], m['completion_tokens'], m['finish_reason']['type']) != (130, 16, 'length'):
            raise ValueError('Prime returned the wrong shape or completion')
    if ids != set(payload['rid']):
        raise ValueError('Prime request IDs differ')
    return (time.perf_counter()-t)*1000


def run(policy):
    def emit(**row):
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    prompts = [f'样本 {i+1}。背景资料（无需复述）：'+('蓝色标签用于标识样本。'*14)
               + f'\n写一段至少两百字的中文，解释{c}色灯光如何帮助人们识别夜间路标。'
               for i, c in enumerate(COLORS)]
    mark('single-warmup')
    for prompt in prompts[:2]:
        call(prompt, max_tokens=24)
    mark('prime')
    prime_ms = native_prime() if policy == 'primed' else None
    emit(kind='policy', policy=policy, prime_wall_ms=prime_ms)
    mark('timed')
    rows = []
    # Fix arrivals in 12 waves of two. Intentional idle while the next wave is built
    # is included in wall time; never compare this rate directly with a sliding pool.
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for wave in range(12):
            rendezvous = threading.Barrier(2, timeout=5)

            def request(i):
                rendezvous.wait()
                return call(prompts[i], max_tokens=48)

            pair = list(pool.map(request, [(2*wave) % 8, (2*wave+1) % 8]))
            for lane, row in enumerate(pair):
                if not (row['done'] and row['finish'] == 'length' and row['completion_tokens'] == 48
                        and row['prompt_tokens'] in (130, 131)):
                    raise ValueError(f'Benchmark request did not meet the fixed contract: {row!r}')
                rows.append(row)
                emit(kind='request', wave=wave, lane=lane, text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                     **{k:v for k,v in row.items() if k != 'text'})
    elapsed = time.perf_counter() - start
    mark('complete')
    emit(kind='summary', **summarize_block(rows, elapsed))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('control','primed'):
        raise SystemExit('Usage: warm_chat_probe.py control|primed')
    run(sys.argv[1])
