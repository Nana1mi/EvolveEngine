"""Diagnostic batch-invariance check on actual Spark chat-template token inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from synthetic_baseline import emit, mark, open_json
from synthetic_baseline_audit import PREFILL

PROMPTS = {
    'english': 'Reply with the single word READY.',
    'chinese': '请只回复这两个汉字：就绪。不要加引号或解释。',
    'long': 'Reference padding; ignore it: ' + ' blue'*740 +
            '\nEnd of padding. Reply with the single word READY.',
}
PLAN = (
    ('english',), ('chinese',), ('long',),
    ('english', 'chinese'), ('chinese', 'english'), ('long', 'english'),
    ('long', 'english'), ('chinese', 'english'), ('english', 'chinese'),
    ('long',), ('chinese',), ('english',),
)
ROUNDS = ('det1', 'det2')


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def validate_finish(meta, output):
    finish = meta['finish_reason']
    count = meta['completion_tokens']
    if not isinstance(finish, dict) or finish.get('type') not in ('stop', 'length'):
        raise ValueError(f'Invalid stop/length result {finish!r}')
    if type(count) is not int or not 0 < count <= 24 or not isinstance(output, list) or not output:
        raise ValueError('Missing generated output')
    if not all(type(i) is int and 0 <= i < 131072 for i in output) or count-len(output) not in (0, 1):
        raise ValueError('Invalid token IDs / count mismatch')
    if finish['type'] == 'length' and count != len(output):
        raise ValueError('Missing length-capped tokens')
    return finish, count


def run(label):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/model', trust_remote_code=True, local_files_only=True)
    template = Path('/model/chat_template.jinja').read_text(encoding='utf-8')
    prompts = {name: tokenizer.encode(tokenizer.apply_chat_template(
        [dict(role='user', content=prompt)], chat_template=template, tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        for name, prompt in PROMPTS.items()}
    emit(kind='det_identity', label=label,
         inputs={name: dict(length=len(ids), sha256=digest(ids)) for name, ids in prompts.items()})
    for n, wave in enumerate(PLAN):
        rids = [f'{label}-wave{n}-{i}' for i in range(len(wave))]
        ids = [prompts[name] for name in wave]
        mark(f'deterministic wave-{n}')
        with open_json('/generate', dict(rid=rids if len(ids) > 1 else rids[0],
                       input_ids=ids if len(ids) > 1 else ids[0], stream=False,
                       sampling_params=dict(temperature=0, max_new_tokens=24))) as response:
            if response.status != 200 or response.headers.get_content_type() != 'application/json':
                raise RuntimeError('Expected successful native JSON response')
            rows = json.load(response)
        if isinstance(rows, dict):
            rows = [rows]
        if len(rows) != len(wave):
            raise ValueError('Wrong batch response size')
        for lane, (row, name) in enumerate(zip(rows, wave)):
            meta, output = row['meta_info'], row['output_ids']
            if (meta['id'], meta['prompt_tokens']) != (rids[lane], len(prompts[name])):
                raise ValueError('Wrong id or prompt count')
            finish, count = validate_finish(meta, output)
            emit(kind='det_reply', wave=n, lane=lane, name=name, output_ids=output,
                 output_sha256=digest(output), finish=finish, completion_tokens=count)
    mark('deterministic complete')


def audit(root):
    by_prompt = defaultdict(lambda: defaultdict(list))
    inputs = None
    prefill_waves = 0
    for label in ROUNDS:
        records = [json.loads(line) for line in (root/f'{label}.stdout').read_text(encoding='utf-8').splitlines()
                   if line.startswith('{')]
        identity, *replies = records
        if (identity.get('kind'), identity.get('label')) != ('det_identity', label):
            raise ValueError(f'{label}: incorrect run identity')
        if inputs is None:
            inputs = identity['inputs']
        rows = []
        logs = (root/f'{label}.server.log').read_text(encoding='utf-8')
        args = [line for line in logs.splitlines() if 'server_args={' in line]
        if not args or not all(s in args[0] for s in ("'enable_deterministic_inference': True", "'attention_backend': 'triton'")):
            raise ValueError(f'{label}: missing effective deterministic/Triton configuration evidence')
        for line in logs.splitlines():
            if line.startswith('EE_CALIBRATION deterministic '):
                rows.append((line.split()[-1], []))
            elif rows and (m := PREFILL.search(line)):
                rows[-1][1].append(tuple(map(int, m.groups())))
        if rows[-1:] != [('complete', [])] or len(rows) != len(PLAN)+1:
            raise ValueError(f'{label}: missing wave isolation and completion markers')
        expected_replies = []
        if identity['inputs'] != inputs:
            raise ValueError('Prompt input changed across modes/restarts')
        for i, wave in enumerate(PLAN):
            expected = (f'wave-{i}', [(len(wave), sum(inputs[k]['length'] for k in wave), 0)])
            if rows[i] != expected:
                raise ValueError(f'{label}: wrong scheduler batch in wave {i}: {rows[i]}, expected {expected}')
            prefill_waves += 1
            expected_replies.extend((i, lane, name, len(wave)) for lane, name in enumerate(wave))
        if len(replies) != len(expected_replies):
            raise ValueError(f'{label}: missing replies')
        for row, (wave, lane, name, width) in zip(replies, expected_replies):
            if (row['kind'], row['wave'], row['lane'], row['name']) != ('det_reply', wave, lane, name):
                raise ValueError(f'{label}: wrong reply identity')
            if row['output_sha256'] != digest(row['output_ids']):
                raise ValueError(f'{label}: wrong output checksum')
            finish, count = validate_finish(dict(finish_reason=row['finish'], completion_tokens=row['completion_tokens']),
                                            row['output_ids'])
            signature = (tuple(row['output_ids']), count, json.dumps(finish, sort_keys=True))
            by_prompt[name][signature].append(dict(run=label, wave=wave, lane=lane, width=width))
    results = {name: [dict(output_sha256=digest(ids), completion_tokens=count, finish=json.loads(finish), sampled_at=seen)
                      for (ids, count, finish), seen in signatures.items()] for name, signatures in by_prompt.items()}
    return dict(passed=all(len(signatures)==1 for signatures in results.values()), inputs=inputs,
                verified_prefill_waves=prefill_waves, verified_requests=sum(len(places) for g in by_prompt.values() for places in g.values()),
                output_groups=results, scope='bounded diagnostic config; no claim for other batches, modes, kernels or inputs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('verb', choices=(*ROUNDS, 'audit'))
    p.add_argument('root', nargs='?', type=Path)
    args = p.parse_args()
    if args.verb == 'audit':
        if args.root is None:
            p.error('audit requires a run directory')
        result = audit(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(not result['passed'])
    if args.root is not None:
        p.error('run does not accept a run directory')
    run(args.verb)


if __name__ == '__main__':
    main()
