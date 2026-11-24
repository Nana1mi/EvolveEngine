"""Offline replay of the G1 r1 exact-reference controls, not a general admission gate."""
import argparse
import base64
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_pair_audit import checked_manifest
from runtime_identity import compare, read_projection

BUNDLE = '5c0c6ee5797148fd5830cd69a7ac3b0367a626d707bb925699be8db568d26cf9'
# Derived offline by applying the pinned reference edit and entry.sh's fault edit to E0.
FAULT = '62d09a07b7f84053428540e04ef1a2f1f533b64f58571bcf187e02c95d2fab6f'
ROLES = ('e0a', 'e0b', 'qkv', 'fault')
KINDS = ('identity', 'before0', 'before1', 'cancel', 'after0', 'after1', 'numeric130', 'numeric260')
VOCAB = 131072
PASS, FAIL, UNKNOWN = 'pass_scoped', 'verified_difference', 'insufficient_evidence'


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def checked_run(root):
    need(sha256((root/'bundle.sha256').read_bytes()).hexdigest() == BUNDLE, 'not the committed bundle')
    checked_manifest(root, 'bundle.sha256', set('PLAN.md plan.json model.sha256 entry.sh node.sh probe.py reference_qkv.py'.split()))
    # Validate independent boots separately: a missing candidate file must not erase
    # verified evidence of a different fault in another boot.
    names = {r+'.'+ext for r in ROLES for ext in ('jsonl', 'server.log', 'pre.csv')}
    index={}; issues={}; verified=0
    for line in (root/'raw.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        need(name in names and name not in index and re.fullmatch('[0-9a-f]{64}',digest), 'invalid raw manifest')
        index[name]=digest
    for name in sorted(names):
        try:
            need(name in index and sha256((root/name).read_bytes()).hexdigest()==index[name], 'missing digest or checksum mismatch: '+name)
            verified+=1
        except (ValueError,OSError) as err:
            issues[name.split('.')[0]]=str(err)
    plan = json.loads((root/'plan.json').read_bytes())
    need(plan['labels'] == list(ROLES), 'role ordering')
    model_files = [s.split('  ')[1] for s in (root/'model.sha256').read_text().splitlines()]
    verify = (root/'model.verify').read_text().splitlines()
    need(verify[:-1] == [s+': OK' for s in model_files] and verify[-1].endswith('/Spark-X2.6-1.7B-Q8_0.gguf: OK'), 'model verification')
    order = [s.split('\t') for s in (root/'run-order.tsv').read_text().splitlines()]
    need([r[:2] for r in order] == [[stage, role] for role in ROLES for stage in ('start', 'done')], 'start/done sequence')
    epochs = [int(r[2]) for r in order]
    need(epochs == sorted(epochs) and all(len(r)==4 and datetime.fromisoformat(r[3]).timestamp()==t for r,t in zip(order, epochs)), 'timestamps')
    closed, rc = (root/'run.closed').read_text().strip().split('\t')
    need(rc=='0' and datetime.fromisoformat(closed).timestamp()>=epochs[-1], 'unclean shutdown')
    need(not (root/'post-compute.csv').read_text().strip() and 'NVIDIA GB10' in (root/'post-gpu.csv').read_text(), 'teardown')
    ref = None; field_count = None
    for role in ROLES:
        if role in issues:
            continue
        p = root/(role+'.server.log')
        try:
            marker = [s for s in p.read_text(encoding='utf-8').splitlines() if s.startswith('EE_G1_SOURCE ')]
            digest = FAULT if role=='fault' else plan['qkv' if role=='qkv' else 'gguf_py']
            need(marker==[f'EE_G1_SOURCE {role} {digest}'], role+' source')
            cur = read_projection(p)
            paired = compare(ref if ref is not None else cur, cur)
            need(not paired['unobserved_fields'] and not paired['differing_fields'], role+' runtime mismatch')
            cap = cur['scheduler_capacity']
            need((cap['max_running_requests'], cap['max_total_num_tokens'], cap['context_len']) == (4,16384,8192) and cur['startup_args']['disable_radix_cache'] is True, role+' runtime dimensions')
            if ref is None:
                ref=cur
            field_count=len(paired['equal_fields'])
        except (ValueError, KeyError, OSError, TypeError) as err:
            issues[role]=str(err)
    return dict(bundle_sha256=BUNDLE, raw_manifest_sha256=sha256((root/'raw.sha256').read_bytes()).hexdigest(),
                raw_file_count=verified, node_elapsed_s=epochs[-1]-epochs[0],
                boot_elapsed_s={role: epochs[2*i+1]-epochs[2*i] for i,role in enumerate(ROLES)}, runtime_equal_fields=field_count), issues


def rows_for(root, role):
    rows = [json.loads(s) for s in (root/(role+'.jsonl')).read_bytes().splitlines()]
    keys = [x['kind'] for x in rows]
    need(keys == [k for k in KINDS if k in keys], role+' reordered/duplicate/unknown evidence')
    data = dict(zip(keys, rows))
    need(data.get('identity') == dict(kind='identity',label=role,eos=1,vocab=VOCAB), role+' model identity')
    return data


def event(data, kind):
    x = data.get(kind, {})
    need(x.get('ok') is True, kind+': '+str(x.get('error_type', 'missing receipt')))
    return x


def natural(data, role, stage, i):
    x = event(data, stage+str(i))
    need(x['rid']==f'g1-{role}-{stage}-{i}' and x['cap']==48 and re.fullmatch('[0-9a-f]{64}',x['input_sha']), 'natural request identity')
    ids = x['ids']; n = x['n']; finish = x['finish']
    need(type(n) is int and 0<n<=48 and isinstance(ids,list) and n==len(ids) and all(type(v) is int and 0<=v<VOCAB for v in ids), 'incomplete token accounting')
    need(isinstance(finish,dict) and (finish.get('type')=='stop' and 'matched' in finish or finish.get('type')=='length' and n==48 and finish.get('length')==48), 'unclear finish reason')
    return (x['input_sha'], ids, finish)


def cancelled(data, role):
    x = event(data,'cancel')
    need(x['rid']==f'g1-abort-{role}-target' and x['cap']==512 and type(x['trigger']) is int and type(x['last_n']) is int and 4<=x['trigger']<=x['last_n']<512, 'no matching in-flight request')
    need(x['abort_http']==200 and x['done'] is True and x['stream_error'] is None and isinstance(x['finish'],dict) and x['finish'].get('type')=='abort', 'no unambiguous abort terminal')
    return {'trigger_tokens':x['trigger'], 'final_tokens':x['last_n']}


def vector(data, n):
    x = event(data, f'numeric{n}')
    input_hash = sha256(json.dumps([6510+i%37 for i in range(n)]).encode()).hexdigest()
    need((x['count'], x['prefix_sha256'], x['vocab'], x['format'])==(n,input_hash,VOCAB,'little-f32-zlib-base64'), 'numeric input or encoding')
    b = zlib.decompress(base64.b64decode(x['vector'],validate=True))
    need(len(b)==4*VOCAB and sha256(b).hexdigest()==x['sha256'], 'full vector length or digest')
    v = struct.unpack(f'<{VOCAB}f', b)
    need(all(math.isfinite(w) and w<=0 for w in v), 'nonfinite or positive log probability')
    mass = math.fsum(map(math.exp,v))
    # Serialization sanity only; NOT a KL, model-drift or acceptance threshold.
    need(abs(mass-1)<0.001, 'not a normalized full distribution')
    return v, mass, x['sha256']


def outcome(equal, **info):
    return dict(status=PASS if equal else FAIL, **info)


def evaluate(fn, *args):
    try:
        return fn(*args)
    except (KeyError, ValueError, TypeError, OSError, OverflowError, zlib.error, IndexError) as err:
        return dict(status=UNKNOWN, reason=str(err))


def eos_check(data, role, stage, i):
    baseline = natural(data['e0a'], 'e0a', 'before', i)
    need(baseline==natural(data['e0b'], 'e0b', 'before', i), 'E0 repeat differs')
    prefix, seq, finish = baseline
    need(finish=={'type':'stop','matched':1} and seq[-1]==1 and len(seq)<48, 'E0 did not establish natural EOS')
    inp, actual, why = natural(data[role],role,stage,i)
    need(inp==prefix, 'different input')
    return outcome((actual,why)==(seq,finish), tokens=len(actual), finish=why['type'], matched=why.get('matched'))


def isolation(data, role, i):
    ack = cancelled(data[role],role)
    before = natural(data[role],role,'before',i); after = natural(data[role],role,'after',i)
    need(before[0]==after[0], 'different sentinel inputs')
    # Within-engine effect only: a consistently wrong sentinel is NOT evidence of a KV leak.
    return outcome(before==after, **ack)


def numeric_check(vec, role, n):
    base, bm, bd = vec['e0a', n]; cur, cm, cd = vec[role, n]
    need(bd==vec['e0b',n][2], 'E0 repeat not byte-identical at this position')
    rank = sorted(range(VOCAB),key=cur.__getitem__,reverse=True)[:2]; btop=max(range(VOCAB),key=base.__getitem__)
    kl = math.fsum(math.exp(a)*(a-b) for a,b in zip(base,cur))/bm + math.log(cm/bm)
    return outcome(bd==cd, sha256=cd, mass=cm, kl_e0_to_role=kl, max_abs_logp=max(abs(a-b) for a,b in zip(base,cur)),
                   e0_top1=btop, top1=rank[0], top1_margin=cur[rank[0]]-cur[rank[1]])


def combine(checks):
    states = [c['status'] for c in checks.values()]
    return FAIL if FAIL in states else UNKNOWN if not states or UNKNOWN in states else PASS


def audit(root):
    run, issues = checked_run(root); data={}
    for role in ROLES:
        try:
            if role in issues:
                data[role]={}
            else:
                data[role]=rows_for(root,role)
        except (ValueError,KeyError,TypeError) as err:
            data[role]={}; issues[role]=str(err)
    vectors={}; bad={}
    for role in ROLES:
        for n in (130,260):
            try:
                vectors[role,n] = vector(data[role],n)
            except (ValueError, KeyError, TypeError, OverflowError, zlib.error) as err:
                bad[role,n] = str(err)
    arms={}
    for role in ROLES:
        checks={f'{stage}{i}':evaluate(eos_check,data,role,stage,i) for stage in ('before','after') for i in (0,1)}
        checks.update({f'post_cancel_stability{i}':evaluate(isolation,data,role,i) for i in (0,1)})
        for n in (130,260):
            err=next((bad[r,n] for r in ('e0a','e0b',role) if (r,n) in bad),None)
            checks[f'numeric{n}']=dict(status=UNKNOWN,reason=err) if err is not None else evaluate(numeric_check,vectors,role,n)
        arms[role]=dict(status=combine(checks), checks=checks)
    expected={r:(FAIL if r=='fault' else PASS) for r in ROLES}
    return dict(scope='P1 r1 exact-reference synthetic controls only', formal_g1_signed=False,
                control_pattern_observed=all(arms[r]['status']==s for r,s in expected.items()),
                evidence_complete=all(c['status']!=UNKNOWN for a in arms.values() for c in a['checks'].values()),
                run=run, receipt_issues=issues, arms=arms)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); args=p.parse_args()
    try:
        report=audit(args.root); success=report['control_pattern_observed'] and report['evidence_complete']
    except (ValueError, KeyError, TypeError, IndexError, OSError) as err:
        report=dict(status=UNKNOWN, formal_g1_signed=False, reason=str(err)); success=False
    print(json.dumps(report,indent=2,sort_keys=True)); return 0 if success else 2


if __name__=='__main__':
    raise SystemExit(main())
