"""Audit the archived receipt from one isolated Chat wire conformance run."""
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from reference_pair_audit import checked_manifest

BUNDLE = {'chat_sse_observer.py','chat_sse_smoke.py','chat_sse_smoke_entry.sh'}
RAW = {'wire.stdout','wire.server.log'}
CASES = [('text_stop','stop'), ('text_length','length'), ('reasoning_and_answer','stop'), ('tool_only','tool_calls')]
TIME = ('first_payload_ms','first_answer_ms','last_payload_ms','done_ms')


def audit(root, want_bundle):
    if sha256((root/'bundle.sha256').read_bytes()).hexdigest() != want_bundle:
        raise ValueError('Wrong pre-run bundle SHA-256')
    bundle = checked_manifest(root,'bundle.sha256',BUNDLE)
    raw = checked_manifest(root,'raw-evidence.sha256',RAW)
    if any((root/f).read_text(encoding='utf-8').strip() for f in ('pre-compute.csv','post-compute.csv')):
        raise ValueError('GPU compute list was not empty at pre/post check')
    if (root/'run.closed').read_text(encoding='utf-8').strip().split('\t')[-1] != '0':
        raise ValueError('Chat wire conformance did not close successfully')
    rows = [json.loads(line) for line in (root/'wire.stdout').read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(CASES)+1 or rows[-1] != dict(kind='wire_summary', passed=4, total=4):
        raise ValueError('Incomplete Chat wire conformance summary')
    for r,(name, finish) in zip(rows[:-1],CASES,strict=True):
        if (r.get('kind'),r.get('case'),r.get('finish')) != ('wire_case',name,finish) or r.get('passed') is not True:
            raise ValueError(f'Unexpected or failing Chat case {name}')
        times = [r[k] for k in TIME]
        if not all(type(t) in (float,int) and isfinite(t) and t >= 0 for t in times) or times != sorted(times):
            raise ValueError(f'Invalid Chat metric chronology for {name}')
        if not all(type(r[k]) is int and r[k] > 0 for k in ('completion_tokens','prompt_tokens','payload_frames')):
            raise ValueError(f'Invalid Chat accounting for {name}')
        if not re.fullmatch(r'[a-f0-9]{64}',r['response_sha256']):
            raise ValueError(f'Invalid Chat semantic digest for {name}')
        if name=='text_length' and r['completion_tokens']!=1:
            raise ValueError('Token cap was not honored')
        if name=='reasoning_and_answer' and not times[0]<times[1]:
            raise ValueError('Reasoning and first answer were not distinguished')
    log = (root/'wire.server.log').read_text(encoding='utf-8')
    if ('runtime /patched/python/sglang/__init__.py ' not in log or
        len(re.findall(r'127\.0\.0\.1:\d+ - "POST /v1/chat/completions HTTP/1\.1" 200 OK',log)) != len(CASES)):
        raise ValueError('Missing pinned-source or loopback route evidence')
    return dict(pass_count=4, scope='Isolated diagnostic E0, synthetic wire conformance only',
                bundle_sha256=want_bundle, verified_bundle=bundle, verified_raw=raw, case_receipts=rows[:-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--bundle-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.bundle_sha256),sort_keys=True,indent=2))


if __name__=='__main__':
    main()
