"""Independent, offline replay of G1-D3. No sealed positions or GPU required."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_d2 import audit as d2
from g1_d2.probe import ih, prefix
from g1_d3.screen import COUNT, score
from reference_pair_audit import checked_manifest
from runtime_identity import compare, read_projection

BUNDLE = '27c99e80bfc40ad4c231204b2d70d94d10da4511fc85c054aaf9548b32d1195e'
ROLES = ('anchor', 'local0a', 'local0b')
FILES = set('entry.sh model.sha256 reference_qkv.py probe.py PLAN.md screen.py variant.py node.sh plan.json'.split())
MODULE = '/patched/python/sglang/srt/layers/quantization/gguf.py'
POSITIONS = range(COUNT)
sha = d2.checksum
need = d2.need


def read_run(root):
    need(sha(root/'bundle.sha256') == BUNDLE, 'preregistration mismatch')
    committed = checked_manifest(root, 'bundle.sha256', FILES)
    need(committed['screen.py'] == sha(Path(__file__).with_name('screen.py')), 'screen changed since registration')
    need(committed['probe.py'] == sha(ROOT/'g1_d2/probe.py'), 'development inputs differ from D2')
    plan = json.loads((root/'plan.json').read_bytes())
    need(plan['labels'] == list(ROLES) and plan['source']['local0a'] == plan['source']['local0b'], 'role identity')
    expected = {role+'.'+suffix for role in ROLES for suffix in ('jsonl','server.log','pre.csv')}
    seen, invalid = set(), {}
    for line in (root/'raw.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        need(name in expected and name not in seen and re.fullmatch('[0-9a-f]{64}',digest), 'ambiguous manifest')
        seen.add(name)
        try:
            need(sha(root/name) == digest, 'checksum mismatch: '+name)
        except (OSError,ValueError) as e:
            invalid[name.split('.')[0]] = str(e)
    for name in sorted(expected-seen):
        invalid[name.split('.')[0]] = 'missing raw manifest entry '+name
    order = [l.split('\t') for l in (root/'run-order.tsv').read_text().splitlines()]
    need([r[:2] for r in order] == [[event,role] for role in ROLES for event in ('start','done')], 'boot order')
    times = [int(r[2]) for r in order]
    need(times == sorted(times) and all(len(r)==4 and datetime.fromisoformat(r[3]).timestamp()==t for r,t in zip(order,times)), 'chronology')
    closed, rc = (root/'run.closed').read_text().strip().split('\t')
    need(rc == '0' and datetime.fromisoformat(closed).timestamp() >= times[-1], 'unclean termination')
    need(not (root/'post-compute.csv').read_text().strip() and 'NVIDIA GB10' in (root/'post-gpu.csv').read_text(), 'GPU teardown')
    files = [l.split('  ')[1] for l in (root/'model.sha256').read_text().splitlines()]
    verified = (root/'model.verify').read_text().splitlines()
    need(verified[:-1] == [s+': OK' for s in files] and verified[-1].endswith('/Spark-X2.6-1.7B-Q8_0.gguf: OK'), 'model verification')
    return plan, invalid, dict(bundle_sha256=BUNDLE, raw_manifest_sha256=sha(root/'raw.sha256'),
                               screen_sha256=committed['screen.py'], node_seconds=times[-1]-times[0],
                               per_boot_s={r:times[2*i+1]-times[2*i] for i,r in enumerate(ROLES)}, closed=closed)


def layer_coverage(lines, role, source):
    tag_to_i = {f'dev-{role}-{i:02}:'+ih(prefix(i)):i for i in POSITIONS}
    groups = defaultdict(Counter)
    for line in lines:
        if not line.startswith('EE_G1D3_HIT '):
            continue
        _, tag, m, layer, mode, path, digest = line.split()
        need(tag in tag_to_i and path == MODULE and digest == source, 'unbound runtime provenance')
        i = tag_to_i[tag]
        need(m == str(len(prefix(i))), 'wrong batch shape')
        groups[i][layer,mode] += 1
    if role == 'anchor':
        need(not groups, 'unexpected instrumented baseline')
        return set(POSITIONS)
    expected = Counter((f'model.layers.{i}.self_attn.q_k_v_proj','split' if i==0 else 'whole') for i in range(28))
    return {i for i in POSITIONS if groups[i] == expected}


def numeric_rows(root, role):
    records = [json.loads(line) for line in (root/(role+'.jsonl')).read_bytes().splitlines()]
    need(all(r.get('kind')=='numeric' and type(r['i']) is int and r['i'] in POSITIONS for r in records), 'unrecognized receipt')
    idx = [r['i'] for r in records]
    need(idx == sorted(set(idx)), 'duplicate/reordered position')
    return {r['i']:r for r in records}


def checked_role(root, role, plan, runtime_ref):
    path = root/(role+'.server.log')
    lines = path.read_text(encoding='utf8').splitlines()
    src = plan['source'][role]
    need([l for l in lines if l.startswith('EE_G1D3_SOURCE ')] == [f'EE_G1D3_SOURCE {role} {src}'], 'source hash mismatch')
    delta = compare(runtime_ref, read_projection(path))
    need(not delta['differing_fields'] and not delta['unobserved_fields'], 'runtime projection mismatch')
    return numeric_rows(root,role), layer_coverage(lines,role,src), len(delta['equal_fields'])


def audit(root, prior):
    plan, issues, run = read_run(root)
    need(sha(prior/'raw.sha256') == plan['prior_raw_manifest'] and sha(prior/'bundle.sha256') == plan['prior_bundle'], 'wrong development archive')
    old_plan,_,bad = d2.loadrun(prior)
    need(not bad and all(plan[k]==old_plan[k] for k in ('image','weight','gguf_py','qkv','context_length','max_total_tokens','max_running_requests')), 'D2 source data or config mismatch')
    old = d2.audit(prior)
    need(old['exact_reference_reproduced'] and not old['issues'], 'historical reference not verified')
    old_e0 = [r for r in (json.loads(s) for s in (prior/'e0a.jsonl').read_bytes().splitlines()) if r['kind']=='numeric']
    ref = read_projection(prior/'e0a.server.log')
    rows, hits, common = {}, {}, {}
    for role in ROLES:
        if role not in issues:
            try:
                rows[role], hits[role], common[role] = checked_role(root,role,plan,ref)
            except (ValueError, KeyError, OSError, TypeError, IndexError) as e:
                issues[role] = str(e)
    arms = {r:{} for r in ROLES[1:]}; errors = []; anchored = []; repeated = []
    for i in POSITIONS:
        vectors = {}
        for role in ROLES:
            if role not in rows:
                continue
            try:
                need(i in hits[role], '28-layer execution not established')
                vectors[role] = d2.unpack(rows[role][i],role,i)
            except (ValueError,KeyError,TypeError,zlib.error) as e:
                errors.append(f'{role}/{i}: {e}')
        before = vectors.get('anchor')
        if before is None or not np.array_equal(before,d2.unpack(old_e0[i],'e0a',i)):
            errors.append(f'anchor/{i}: missing or differs from D2');continue
        anchored.append(i)
        for role in arms:
            if role in vectors:
                arms[role][i] = d2.metrics(before,vectors[role])
        if all(role in vectors for role in arms) and np.array_equal(vectors['local0a'],vectors['local0b']):
            repeated.append(i)
    def summary(arm):
        vs=list(arm.values());return dict(n=len(vs), exact=sum(m['exact'] for m in vs), flips=sum(m['flip'] for m in vs),
            metrics={k:dict(min=min(m[k] for m in vs),median=float(np.median([m[k] for m in vs])),max=max(m[k] for m in vs)) for k in ('tv','kl','top10_max','full_max')} if vs else {})
    historical = {role:{p['i']:p['arms'][role] for p in old['positions']} for role in ('e0b','b','c','d')}
    screens={r:score(m) for r,m in arms.items()}
    return dict(scope='G1-D3 development only', formal_g1=False, validation_opened=False, run=run,
                boot_issues=issues, position_issues=errors, anchor_exact_vs_d2=len(anchored),
                verified_all_28_layers={r:len(hits.get(r,set())) for r in arms}, equal_runtime_fields=common,
                exact_candidate_repeat_positions=len(repeated), summaries={r:summary(m) for r,m in arms.items()},
                prospective_screens=screens, retrospective_d2_screens={r:score(m) for r,m in historical.items()},
                positions={r:{str(i):m for i,m in arm.items()} for r,arm in arms.items()})


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('raw',type=Path);cli.add_argument('d2_raw',type=Path);args=cli.parse_args()
    try:
        data=audit(args.raw,args.d2_raw)
        complete=not data['boot_issues'] and not data['position_issues'] and data['anchor_exact_vs_d2']==COUNT and data['exact_candidate_repeat_positions']==COUNT
    except (ValueError,OSError,KeyError,TypeError,IndexError) as e:
        data={'scope':'G1-D3','status':'insufficient_evidence','reason':str(e),'formal_g1':False};complete=False
    print(json.dumps(data,indent=2,sort_keys=True,allow_nan=False));sys.exit(0 if complete else 2)
