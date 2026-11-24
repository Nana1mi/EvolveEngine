"""Local-only API checks and small synthetic repeated-run measurements."""
import concurrent.futures
import json
import math
import statistics
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

ENDPOINT = 'http://127.0.0.1:30091/v1/chat/completions'


def call(prompt, *, stream=True, max_tokens=32, **extra):
    payload = dict(model='spark26-q8', messages=[dict(role='user', content=prompt)],
                   chat_template_kwargs=dict(enable_thinking=False), temperature=0,
                   stream=stream, max_tokens=max_tokens, **extra)
    if stream:
        payload['stream_options'] = dict(include_usage=True)
    request = Request(ENDPOINT, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    opener = build_opener(ProxyHandler({}))
    start, first, last = time.perf_counter(), None, None
    done, finish, usage, chunks = False, None, {}, []
    with opener.open(request, timeout=45) as response:
        if response.status != 200:
            raise ValueError(f'HTTP {response.status}')
        if stream:
            if response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('Expected SSE')
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise ValueError(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', ()):
                    content = choice.get('delta', {}).get('content')
                    if content:
                        last = time.perf_counter()
                        first = last if first is None else first
                        chunks.append(content)
                    finish = choice.get('finish_reason') or finish
            if not done or finish not in ('stop', 'length') or first is None:
                raise ValueError(f'Incomplete SSE: done={done}, finish={finish}, content={chunks!r}')
        else:
            event = json.load(response)
            usage = event['usage']
            finish = event['choices'][0]['finish_reason']
            chunks.append(event['choices'][0]['message']['content'])
    if not all(isinstance(usage.get(key), int) and usage[key] > 0
               for key in ('prompt_tokens', 'completion_tokens')):
        raise ValueError(f'Missing or invalid token usage: {usage!r}')
    count = usage['completion_tokens']
    return dict(text=''.join(chunks), finish=finish, done=done, completion_tokens=count,
                prompt_tokens=usage['prompt_tokens'], content_chunks=len(chunks),
                e2e_ms=(time.perf_counter()-start)*1000,
                ttft_ms=None if first is None else (first-start)*1000,
                tpot_ms=None if first is None or count < 2 else (last-first)*1000/(count-1))


def emit(**data):
    print(json.dumps(data, ensure_ascii=False, sort_keys=True), flush=True)


def quality():
    def exact(response, expected):
        return response['text'].strip().rstrip('.。') == expected and response['finish'] == 'stop'

    simple = 'Reply with the single word READY.'
    chinese = '请只回复这两个汉字：就绪。不要加引号或解释。'
    near = ('Reference padding; ignore it: ' + ' blue'*740 +
            '\nEnd of padding. Reply with the single word READY.')
    cases = [
        ('english_stream', lambda: call(simple), lambda r: exact(r, 'READY')),
        ('chinese_stream', lambda: call(chinese), lambda r: exact(r, '就绪')),
        ('chinese_nonstream', lambda: call(chinese, stream=False), lambda r: exact(r, '就绪')),
        ('explicit_stop', lambda: call(simple, stop=['ADY']),
         lambda r: r['text'].strip() == 'RE' and r['finish'] == 'stop'),
        ('token_cap', lambda: call(simple, stream=False, max_tokens=1),
         lambda r: r['finish'] == 'length' and r['completion_tokens'] == 1),
        ('bounded_longer_prompt', lambda: call(near),
         lambda r: 700 <= r['prompt_tokens'] <= 960 and exact(r, 'READY')),
    ]
    passed = 0
    for name, request, oracle in cases:
        try:
            reply = request()
            ok = bool(oracle(reply))
            emit(kind='quality', case=name, passed=ok, observed=reply)
            passed += ok
        except Exception as exc:
            emit(kind='quality', case=name, passed=False, error=f'{type(exc).__name__}: {exc}')
    emit(kind='quality_summary', passed=passed, total=len(cases))
    return 0 if passed == len(cases) else 1


def percentile(values, q):
    """Nearest-rank sample percentile, not a population latency estimate."""
    if not values or not 0 < q <= 1:
        raise ValueError('Expected values and quantile in (0,1]')
    return sorted(values)[math.ceil(q * len(values)) - 1]


def summarize_block(requests, elapsed_s):
    ttft = [req['ttft_ms'] for req in requests]
    tpot = [req['tpot_ms'] for req in requests if req['tpot_ms'] is not None]
    return dict(count=len(requests), output_tokens=sum(r['completion_tokens'] for r in requests),
                output_tok_s=sum(r['completion_tokens'] for r in requests)/elapsed_s,
                requests_s=len(requests)/elapsed_s, wall_s=elapsed_s,
                prompt_tokens_min=min(r['prompt_tokens'] for r in requests),
                prompt_tokens_max=max(r['prompt_tokens'] for r in requests),
                ttft_p50_ms=statistics.median(ttft), ttft_p95_ms=percentile(ttft, .95),
                tpot_p95_ms=percentile(tpot, .95) if tpot else None)


def bench(order):
    # Eight prescribed synthetic requests per block. Radix reuse is disabled at the server.
    colors = ('红', '蓝', '绿', '白', '橙', '灰', '银', '黑')
    prompts = [f'样本 {i+1}。背景资料（无需复述）：'+('蓝色标签用于标识样本。'*14)
               + f'\n写一段至少两百字的中文，解释{c}色灯光如何帮助人们识别夜间路标。'
               for i, c in enumerate(colors)]
    for concurrency in order:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Warm-up is excluded, done at the matching concurrency before each block.
            list(pool.map(lambda p: call(p, max_tokens=24), prompts[:concurrency]))
            t0 = time.perf_counter()
            results = list(pool.map(lambda p: call(p, max_tokens=48), prompts))
            elapsed_s = time.perf_counter() - t0
        for i, result in enumerate(results):
            # Do not emit the generated text in the timing logs.
            emit(kind='bench_request', concurrency=concurrency, index=i,
                 **{key: val for key, val in result.items() if key != 'text'})
        emit(kind='bench_block', concurrency=concurrency, **summarize_block(results, elapsed_s))
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ['quality']:
        raise SystemExit(quality())
    if sys.argv[1:] in (['bench'], ['bench', 'reverse']):
        raise SystemExit(bench((2, 1) if len(sys.argv) == 3 else (1, 2)))
    raise SystemExit('Usage: e0_probe.py quality|bench [reverse]')
