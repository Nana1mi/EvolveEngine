"""Typed, source-identified performance knobs. Never interpret an Agent-provided argv."""
import json
import math
import re

PATH = 'ee_runtime.json'
VERSION = 'typed-performance-argv-v1'
POSITIVE = {'max-running-requests','max-total-tokens','cuda-graph-max-bs-decode','speculative-num-draft-tokens',
            'speculative-num-steps','speculative-ngram-max-trie-depth','speculative-ngram-capacity',
            'speculative-ngram-min-bfs-breadth','speculative-ngram-max-bfs-breadth'}
BUCKETS = {'cuda-graph-bs-decode','cuda-graph-bs-prefill'}


def options(raw):
    def unique(pairs):
        d = dict(pairs)
        if len(d)!=len(pairs): raise ValueError('duplicate runtime option')
        return d
    if type(raw) is not bytes or not 0 < len(raw)<=16384: raise ValueError('invalid runtime declaration size')
    row=json.loads(raw.decode('utf-8'),object_pairs_hook=unique)
    if type(row) is not dict: raise ValueError('expected runtime options object')
    for key,value in row.items():
        if key=='radix-cache': ok=type(value) is bool
        elif key in POSITIVE: ok=type(value) is int and 0<value<=2**31-1
        elif key=='chunked-prefill-size': ok=type(value) is int and (value==-1 or 0<value<=2**31-1)
        elif key=='mem-fraction-static': ok=type(value) in (float,int) and math.isfinite(value) and 0<value<1
        elif key in BUCKETS: ok=(type(value) is list and value and all(type(n) is int and 0<n<=2**31-1 for n in value)
                                 and sorted(set(value))==value)
        elif key in ('attention-backend','schedule-policy'): ok=type(value) is str and re.fullmatch('[a-z][a-z0-9_-]{0,47}',value)
        elif key=='speculative-algorithm': ok=value=='NGRAM'  # no extra weights, paths or remote dependency
        else: ok=False
        if not ok: raise ValueError('invalid or locked runtime option: '+key[:64])
    return row


def apply(base, raw):
    if not raw: return tuple(base)
    opts=options(raw); argv=list(base)
    for key,v in sorted(opts.items()):
        if key=='radix-cache':
            if '--disable-radix-cache' in argv: argv.remove('--disable-radix-cache')
            if not v: argv.append('--disable-radix-cache')
            continue
        flag='--'+key
        if flag in argv:
            idx=argv.index(flag);del argv[idx:idx+2]
        argv.extend((flag,*(str(n) for n in (v if type(v) is list else [v]))))
    return tuple(argv)
