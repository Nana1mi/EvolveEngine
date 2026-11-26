"""Versioned client-side timing of independently registered requests."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from math import isfinite
import json
from pathlib import Path
from threading import Barrier
from time import perf_counter_ns
from urllib.request import Request, ProxyHandler, build_opener
from uuid import uuid4



def fingerprint(ids):
    return sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def merge_stream_output(previous, emitted, completion_tokens, *, final=False):
    """Reconstruct SGLang cumulative or incremental output_ids without gaps."""
    if (type(completion_tokens) is not int or completion_tokens < len(previous)
            or not isinstance(emitted, list) or any(type(token) is not int for token in emitted)):
        raise ValueError('invalid streamed output token sequence')
    delta = completion_tokens - len(previous)
    if delta == 0:
        # Cumulative streams repeat the full prefix; incremental streams may
        # emit an empty terminal frame (or repeat the already-seen final token).
        if emitted == previous or not emitted or (final and emitted == previous[-len(emitted):]):
            return previous
        raise ValueError('stream emitted tokens without advancing completion_tokens')
    if len(emitted) == completion_tokens:
        if emitted[:len(previous)] != previous:
            raise ValueError('cumulative output_ids changed an earlier token')
        return emitted.copy()
    if len(emitted) == delta:
        return previous + emitted
    raise ValueError('output_ids do not match cumulative completion_tokens or incremental delta '
                     f'(confirmed={len(previous)}, reported={completion_tokens}, ids={len(emitted)}, final={final})')


def speculation(meta):
    keys = ('spec_num_correct_drafts', 'spec_num_proposed_drafts', 'spec_verify_ct', 'spec_accept_rate')
    if not any(key in meta for key in keys):
        return None
    accepted, proposed, verifies, rate = (meta.get(key) for key in keys)
    if (not all(type(x) is int for x in (accepted, proposed, verifies)) or not 0 <= accepted <= proposed
            or proposed < 1 or verifies < 1 or type(rate) not in (float, int) or not isfinite(rate)
            or abs(rate - accepted / proposed) > 1e-9):
        raise ValueError('invalid server-reported speculative metrics')
    return dict(accepted=accepted, proposed=proposed, verifies=verifies)


def request(ids, out, gate, *, client=None, clock=perf_counter_ns, natural_eos=False, answer_codec=None, retain_ids=False, quality_only=False):
    rid = uuid4().hex
    n = len(ids)
    body = dict(rid=rid, input_ids=ids, stream=True,
                sampling_params=dict(temperature=0, ignore_eos=not natural_eos, max_new_tokens=out))
    message = Request('http://127.0.0.1:30091/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    client = build_opener(ProxyHandler({})) if client is None else client
    gate.wait(timeout=10)
    start = clock()
    first = last = finish = done = None
    answer = reasoning_end = None
    count = 0
    output_pin = spec_metrics = reason = None
    reconstructed = []
    pending = []  # unconfirmed cumulative prefix on non-timed, quality-only E0 streams
    first_is_single = False
    with client.open(message, timeout=100) as reply:
        if reply.status != 200 or reply.headers.get_content_type() != 'text/event-stream':
            raise ValueError('invalid timing SSE status or content type')
        while line := reply.readline(1 << 20):
            if len(line) == (1 << 20):
                raise ValueError('oversized timing SSE frame')
            if not line.startswith(b'data:'):
                continue
            timestamp = clock()
            data = line[5:].strip()
            if done is not None:
                raise ValueError('unexpected SSE data after DONE')
            if data == b'[DONE]':
                done = timestamp
                if retain_ids:
                    continue  # consume to EOF; no trailing data events are legal
                break
            row = json.loads(data)
            meta = row['meta_info']
            produced = meta['completion_tokens']
            if (finish is not None or meta['id'] != rid or meta['prompt_tokens'] != n
                    or type(produced) is not int or not count <= produced <= out):
                raise ValueError('invalid timing stream identity or token count')
            termination = meta.get('finish_reason')
            emitted = row.get('output_ids')
            # E0's non-incremental interim frame can reference a mutable list:
            # serialized IDs may lead that frame's count. Quarantine the preview;
            # every token must later be witnessed at the corresponding count.
            # This never licenses a fabricated EOS, a missing delta, or a TTFT.
            single_frame = produced == 1 and isinstance(emitted, list) and len(emitted) == 1
            if quality_only and natural_eos and retain_ids and isinstance(emitted, list) and len(emitted) > produced:
                if (termination is not None or len(emitted) > out or emitted[:len(reconstructed)] != reconstructed
                        or emitted[:min(len(pending),len(emitted))] != pending[:min(len(pending),len(emitted))]
                        or not all(type(t) is int and 0 <= t < 131072 for t in emitted)):
                    raise ValueError('invalid cumulative output preview')
                if len(emitted) > len(pending):
                    pending = emitted.copy()
                emitted = emitted[:produced]
            reconstructed = merge_stream_output(reconstructed, emitted, produced, final=termination is not None)
            if pending:
                common = min(len(reconstructed),len(pending))
                if reconstructed[:common] != pending[:common] or (termination is not None and common < len(pending)):
                    raise ValueError('output preview not confirmed on the same stream')
                if common == len(pending):
                    pending = []
            if produced > count:
                if first is None:
                    first_is_single = single_frame
                    if not quality_only and produced != 1:
                        raise ValueError('first token timing ambiguous')
                    first = timestamp
                last = timestamp
                count = produced
                if answer_codec is not None and answer is None:
                    if reasoning_end is None and 4 in reconstructed:
                        reasoning_end = reconstructed.index(4)
                    if reasoning_end is not None and answer_codec.decode(
                            reconstructed[reasoning_end+1:],skip_special_tokens=True).strip():
                        answer = timestamp
            if termination is not None:
                reason = termination['type']
                normal = (natural_eos and reason == 'stop' and 0 < count <= out
                          and type(termination.get('matched')) is int and termination['matched'] == 1 and reconstructed[-1] == 1)
                if (not normal and (reason != 'length' or count != out)) or len(reconstructed) != count:
                    raise ValueError('invalid timing stream termination')
                output_pin = fingerprint(reconstructed)
                spec_metrics = speculation(meta)
                finish = timestamp
    if not (first and last and finish and done and start < first <= last <= finish <= done
            and (quality_only or (first < last if count > 1 else natural_eos and first == last))):
        raise ValueError('incomplete timing stream')
    result = dict(rid=rid, input=n, output=count, sha256=output_pin, input_sha256=fingerprint(ids), start_ns=start,
                  first_ns=first, last_ns=last, done_ns=done, first_token_unambiguous=first_is_single, stream_complete=True,
                  reported_speculation=spec_metrics)
    if natural_eos:
        result.update(finish_reason=reason, output_cap=out)
    if retain_ids:
        if not natural_eos or any(not 0 <= t < 131072 for t in reconstructed) or 1 in reconstructed[:-1] or (reason == 'stop' and reconstructed[-1] != 1):
            raise ValueError('invalid observed vocabulary or tokens after EOS')
        result['observed_ids'] = reconstructed
        result['observed_finish'] = termination
    if answer_codec is not None:
        result['visible_answer_ns'] = answer
    return result


def main():
    root = Path('/observer')
    plan = json.loads((root / 'workload.json').read_bytes())
    inputs = json.loads((root / 'inputs.json').read_bytes())
    from observe_timing import wait_ready
    wait_ready()
    codec = None
    if plan.get('w3_visible_answer') == 'spark_after_reasoning_end':
        from tokenizers import Tokenizer

        codec = Tokenizer.from_file('/model/tokenizer.json')
        if codec.encode('</think>',add_special_tokens=False).ids != [4]:
            raise ValueError('unexpected Spark reasoning boundary')
    for scene, shape in plan['scenes'].items():
        for turn, wave in enumerate(inputs[scene]):
            bounds = plan.get('w3_input_range') if scene == 'W3' else None
            ranges=plan['lane_ranges'][scene] if 'lane_ranges' in plan else [bounds or (n,n) for n,_ in shape]
            if len(wave)!=len(shape) or len(ranges)!=len(shape) or any(not lo<=len(ids)<=hi for ids,(lo,hi) in zip(wave,ranges)):
                raise ValueError('timing inputs no longer match request shapes')
            with ThreadPoolExecutor(max_workers=len(shape)) as pool:
                gate = Barrier(len(shape))
                jobs = [pool.submit(request, ids, out, gate, natural_eos=plan.get('timing_mode') == 'natural_eos',
                                    answer_codec=codec if scene=='W3' else None, retain_ids=bool(plan.get('replay_protocol')))
                        for ids, (_, out) in zip(wave, shape)]
                results = [job.result() for job in jobs]
            print(json.dumps(dict(scene=scene, index=turn, timed=turn >= plan['warm_waves'], requests=results)), flush=True)


if __name__ == '__main__':
    main()
