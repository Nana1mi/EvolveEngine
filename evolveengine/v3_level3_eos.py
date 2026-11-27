"""Development EOS behavior on the four registered prompts; not an output similarity metric."""
from hashlib import sha256
import json
import re

PROTOCOL = 'natural-eos-stream-v2'
CASES = ('before0', 'before1', 'after0', 'after1')


def semantic_stop(ids, finish):
    # These prompts request a nonempty answer and have a registered natural stop.
    # A fabricated stop after a plausible nonempty prefix needs the separate numeric screen.
    return (type(ids) is list and 1 < len(ids) <= 48 and
            all(type(t) is int and 0 <= t < 131072 for t in ids) and ids.count(1) == 1 and ids[-1] == 1 and
            type(finish) is dict and finish.get('type') == 'stop' and type(finish.get('matched')) is int and finish['matched'] == 1)


def _base_without_mixed(rows,d3,r1,*,d3_manifest=None,r1_manifest=None):
    # Preserve the legacy numeric anchors, abort and prompt identity checks; do not
    # require a single-lane/batched comparison once paired isolation is enabled.
    import audit_worker as j
    base={r['i']:r for r in j.raw_rows(j.trusted_reference(d3,'anchor.jsonl',d3_manifest or j.D3))}
    nr=[r for r in rows if r['kind']=='numeric']
    j.need([r['i'] for r in nr]==list(range(32)), 'missing DEV numeric positions')
    measured={r['i']:j.metrics(j.unpack(base[r['i']],'anchor',r['i']),j.unpack(r,j.ROLE,r['i'])) for r in nr}
    screen=j.score(measured)
    current={r['kind']:r for r in rows if r['kind'] in j.KINDS}
    j.need(set(current)==set(j.KINDS) and len(rows)==32+len(current) and
           current['identity']==dict(kind='identity',label=j.ROLE,eos=1,vocab=131072),'missing EOS identity or cases')
    j.cancelled(current,j.ROLE); j.vector(current,130); j.vector(current,260)
    ref={r['kind']:r for r in j.raw_rows(j.trusted_reference(r1,'e0a.jsonl',r1_manifest or j.R1))}
    matches=0
    for i in range(2):
        expected=j.natural(ref,'e0a','before',i)
        for stage in ('before','after'):
            actual=j.natural(current,j.ROLE,stage,i)
            j.need(actual[0]==expected[0],'natural EOS prompt differs')
            matches+=actual==expected
    return dict(status='pending_trajectory',max_tv=max(v['tv'] for v in measured.values()),screen=screen,
                eos_passed=matches,mixed_passed=None)


def grade(rows, d3, r1, *, paired=False, d3_manifest=None, r1_manifest=None,
          long_ctx_anchor=None, long_ctx_sha256=None):
    # Legacy parser still authenticates the two anchors, the four prompt/rid pairs,
    # the standalone abort, both additional numeric probes and the mixed chronology.
    from audit_worker import quality
    long_ctx_rows = [r for r in rows if r.get('kind') == 'long_context_logprob']
    original = [r for r in rows if r['kind'] not in ('natural_stream', 'long_context_logprob')]
    if paired:
        legacy = _base_without_mixed(original, d3, r1, d3_manifest=d3_manifest, r1_manifest=r1_manifest)
    elif d3_manifest is None and r1_manifest is None:
        legacy = quality(original, d3, r1)
    else:
        legacy = quality(original, d3, r1, d3_manifest=d3_manifest, r1_manifest=r1_manifest)
    if set(legacy) != {'status','max_tv','screen','eos_passed','mixed_passed'}:
        raise ValueError('unregistered development judge result')
    captures = [r for r in rows if r['kind'] == 'natural_stream']
    if [r['case'] for r in captures] != list(CASES):
        raise ValueError('missing or duplicated natural stop streams')
    old = {r['kind']:r for r in original}
    fresh = set()
    observations = {}
    for row in captures:
        case, w = row['case'], row['stream']
        orig = old[case]
        ids = w['observed_ids']
        prompt = row['prompt_ids']
        if not (type(prompt) is list and len(prompt) == w['input'] and
                sha256(json.dumps(prompt).encode()).hexdigest() == orig['input_sha'] and
                sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest() == w['input_sha256'] and
                isinstance(w['rid'], str) and re.fullmatch(r'[0-9a-f]{32}', w['rid']) and w['rid'] not in fresh and
                w['stream_complete'] is True and w['output_cap'] == orig['cap'] == 48 and
                w['output'] == len(ids) and w['sha256'] == sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()):
            raise ValueError('unbound or incomplete EOS stream')
        fresh.add(w['rid'])
        observations[case] = (semantic_stop(orig['ids'],orig['finish']) and w['finish_reason'] == 'stop' and
                              semantic_stop(ids, w['observed_finish']))
    passed = sum(observations.values())
    long_ctx = None
    if long_ctx_anchor is not None:
        from .v3_level3_long_ctx import grade_long_ctx
        long_ctx = grade_long_ctx(long_ctx_rows, long_ctx_anchor, long_ctx_sha256)
    elif long_ctx_rows:
        raise ValueError('unexpected long-context logprob rows without pinned anchor')
    long_ctx_ok = long_ctx is None or long_ctx['status'] == 'valid'
    result = dict(legacy, status='pending_trajectory' if paired else 'valid' if passed == 4 and legacy['screen']['status'] == 'within_dev_screen'
                  and legacy['mixed_passed'] is True and long_ctx_ok else 'violation', eos_passed=passed,
                  eos_contract=PROTOCOL, eos_outcomes=observations, legacy_exact_matches_diagnostic=legacy['eos_passed'])
    if long_ctx is not None:
        result['long_context'] = long_ctx
    return result
