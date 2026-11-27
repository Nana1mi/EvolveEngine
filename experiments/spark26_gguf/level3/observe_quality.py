"""Independent Native quality inputs. Never reads anchor labels or expected outputs."""
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener

import numeric
import json
from secrets import randbits
from uuid import uuid4


def stochastic_samples(plan):
    for case in plan['cases']:
        prefix = numeric.prefix(case)
        results = []
        for start in range(0, plan['samples'], 64):
            rids = [uuid4().hex for _ in range(min(64, plan['samples']-start))]
            params = [dict(temperature=.9, top_p=.9, top_k=40, max_new_tokens=3,
                           ignore_eos=False, sampling_seed=randbits(63)) for _ in rids]
            with numeric.call('generate',dict(rid=rids,input_ids=[prefix for _ in rids],sampling_params=params)) as reply:
                rows = json.load(reply)
            if type(rows) is not list or len(rows) != len(rids):
                raise ValueError('missing sampled requests')
            by_id = {row['meta_info']['id']:row for row in rows}
            if set(by_id) != set(rids):
                raise ValueError('sample request identities differ')
            for rid in rids:
                row = by_id[rid]
                ids = row['output_ids']
                meta = row['meta_info']
                if meta['prompt_tokens'] != len(prefix) or meta['completion_tokens'] != len(ids):
                    raise ValueError('sample token accounting differs')
                results.append(ids)
        numeric.emit('sampling', case=case, sequences=results)


def streaming_prompts():
    from transformers import AutoTokenizer
    model = Path('/model')
    tok = AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
    template = (model/'chat_template.jinja').read_text()
    prompts = ('Reply with the single word READY.', '请只回复这两个汉字：就绪。不要加引号或解释。')
    return [tok.encode(tok.apply_chat_template([dict(role='user',content=p)],chat_template=template,
                      tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for p in prompts]


def natural_streams(prompts, stage):
    from threading import Barrier
    from observe_text_timing import request
    for i, ids in enumerate(prompts):
        w = request(ids, 48, Barrier(1), natural_eos=True, retain_ids=True, quality_only=True)
        numeric.emit('natural_stream',case=f'{stage}{i}',prompt_ids=ids,stream=w)


def long_context_logprobs(cases_doc):
    from urllib.request import Request
    client = build_opener(ProxyHandler({}))
    for case_name in ('ctx16k', 'ctx64k'):
        entry = cases_doc['cases'][case_name]
        ids = entry['ids']
        rid = uuid4().hex
        body = dict(
            rid=rid,
            input_ids=ids,
            sampling_params=dict(temperature=0.0, max_new_tokens=1, ignore_eos=True),
            return_logprob=True,
            top_logprobs_num=64,
        )
        req = Request('http://127.0.0.1:30091/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with client.open(req, timeout=240) as reply:
            row = json.load(reply)
        meta = row['meta_info']
        if meta.get('id') != rid or meta.get('prompt_tokens') != len(ids) or meta.get('completion_tokens') != 1:
            raise ValueError(f'long-context logprob identity or token count mismatch for {case_name}')
        out_lp = meta['output_token_logprobs'][0]
        top_lps = meta['output_top_logprobs'][0]
        top_map = {str(int(tok_id)): float(lp) for lp, tok_id, *_ in top_lps}
        numeric.emit(
            'long_context_logprob',
            case=case_name,
            prompt_tokens=len(ids),
            prompt_sha256=entry['prompt_sha256'],
            top1_id=int(out_lp[1]),
            top1_logprob=float(out_lp[0]),
            top_logprobs=top_map,
        )


def main():
    end = monotonic() + 180
    client = build_opener(ProxyHandler({}))
    while monotonic() < end:
        try:
            with client.open('http://127.0.0.1:30091/health',timeout=2) as reply:
                if reply.status == 200:
                    break
        except OSError:
            sleep(.25)
    else:
        raise RuntimeError('no candidate health witness')
    only=Path(__file__).with_name('sampling-only.json')
    if only.is_file():
        stochastic_samples(json.loads(only.read_bytes()))
        return
    print('{"kind":"level3_observer_ready"}',flush=True)
    if sys.stdin.buffer.readline(32) != b'begin\n':
        raise RuntimeError('host did not release the quality-observer start barrier')
    for i in range(32):
        numeric.emit('numeric',ok=True,**numeric.numeric('worker-dev',i))
    streams = Path(__file__).with_name('observe_text_timing.py').is_file()
    if streams:
        # The observer has a 1 GiB cgroup; release the tokenizer process before
        # invoking the standalone, independently pinned legacy numeric client.
        prompts = json.loads(subprocess.check_output([sys.executable, __file__, '--render-eos-prompts']))
        natural_streams(prompts, 'before')
    if not Path(__file__).with_name('paired-trajectory.json').is_file():
        for stage in (0,8,32):
            numeric.emit('kv',stage=stage,ok=True,**numeric.mixed('worker-dev',stage))
    subprocess.run([sys.executable,str(Path(__file__).with_name('legacy_eos.py')),'worker-dev'],check=True)
    if streams:
        natural_streams(prompts, 'after')
    long_ctx = Path(__file__).with_name('long-context-cases.json')
    if long_ctx.is_file():
        long_context_logprobs(json.loads(long_ctx.read_bytes()))
    sampling = Path(__file__).with_name('sampling-policy.json')
    if sampling.is_file():
        stochastic_samples(json.loads(sampling.read_bytes()))


if __name__ == '__main__':
    if sys.argv[1:] == ['--render-eos-prompts']:
        print(json.dumps(streaming_prompts()))
    else:
        main()
