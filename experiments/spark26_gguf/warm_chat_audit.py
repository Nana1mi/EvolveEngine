"""Audit actual shape exposure before comparing the chat prewarm policies."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

TRIALS = (('control1', 'control'), ('primed1', 'primed'), ('primed2', 'primed'), ('control2', 'control'))
PHASES = ('single-warmup', 'prime', 'timed', 'complete')
JOINT = re.compile(r'Prefill batch, #new-seq: 2, #new-token: (\d+),')


def joint_by_phase(log):
    phases = []
    seen = {phase: Counter() for phase in PHASES}
    for line in log.splitlines():
        if line.startswith('EE_CHAT_PHASE '):
            phases.append(line.split()[-1])
            continue
        match = JOINT.search(line)
        if match and phases:
            seen[phases[-1]][int(match[1])] += 1
    if tuple(phases) != PHASES:
        raise ValueError(f'Missing or unordered chat markers: {phases!r}')
    return seen


def audit(root):
    evidence = {}
    texts = []
    for name, policy in TRIALS:
        path = root/f'{name}.stdout'
        rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.startswith('{')]
        if len(rows) != 26 or rows[0].get('kind') != 'policy' or rows[0]['policy'] != policy or rows[-1].get('kind') != 'summary':
            raise ValueError(f'{name}: wrong policy or incomplete run')
        requests = rows[1:-1]
        for i, row in enumerate(requests):
            if ((row['wave'], row['lane']) != divmod(i, 2) or row['finish'] != 'length'
                    or row['completion_tokens'] != 48 or not row['done'] or row['prompt_tokens'] not in (130, 131)):
                raise ValueError(f'{name}: invalid request receipt {row!r}')
        timeline = joint_by_phase((root/f'{name}.server.log').read_text(encoding='utf-8'))
        target_prime = Counter({260: 1}) if policy == 'primed' else Counter()
        if timeline['prime'] != target_prime or timeline['single-warmup']:
            raise ValueError(f'{name}: missing/covert joint prefill before timed phase')
        texts.append([r['text_sha256'] for r in requests])
        evidence[name] = dict(policy=policy, prime_wall_ms=rows[0]['prime_wall_ms'], summary=rows[-1],
                              joint_by_phase=timeline,
                              timed_max_ttft_ms=max(r['ttft_ms'] for r in requests),
                              timed_over_500ms=sum(r['ttft_ms']>500 for r in requests))
    matched = all(evidence[name]['joint_by_phase']['timed'][260] > 0 for name, _ in TRIALS)
    return dict(all_trials_exposed_to_timed_target_joint_shape=matched,
                output_positions_with_variation=sum(len(set(col)) != 1 for col in zip(*texts)),
                trials=evidence)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: warm_chat_audit.py <chat-study-directory>')
    root = Path(sys.argv[1])
    result = audit(root)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    (root/'audited-chat-comparison.json').write_text(serialized+'\n', encoding='utf-8')
    print(serialized)
