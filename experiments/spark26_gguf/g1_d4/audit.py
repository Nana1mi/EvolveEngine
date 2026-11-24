"""Replay G1-D4 raw evidence against the frozen bundle and D3 anchor."""
import argparse
import base64
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import zlib

import probe
from g1d4_hook import bf16


def digest(b):
    return sha256(b).hexdigest()


def check(root, manifest):
    items = {}
    for line in (root / manifest).read_text(encoding='utf8').splitlines():
        h, name = line.split('  ', 1)
        if Path(name).name != name or name in items or digest((root/name).read_bytes()) != h:
            raise ValueError(f'invalid {manifest} member {name}')
        items[name] = h
    return items


def readvec(f):
    out = {}
    for line in f.read_text(encoding='utf8').splitlines():
        v = json.loads(line)
        i = v['i']
        raw = zlib.decompress(base64.b64decode(v['vector'], validate=True))
        if (i in out or not v['ok'] or v['kind'] != 'numeric' or v['input_sha'] != probe.ih(probe.prefix(i))
                or len(raw) != 4 * probe.VOCAB or v['sha'] != digest(raw) or v['n'] != len(probe.prefix(i))):
            raise ValueError(f'invalid full-vocabulary row {i}')
        out[i] = (v, raw)
    return out


def stats(a, b):
    err = [abs(x-y) for x,y in zip(a,b,strict=True)]
    return dict(n=len(err), different=sum(x!=0 for x in err), rmse=math.sqrt(statistics.mean(x*x for x in err)), max_abs=max(err))


def audit(root, previous):
    frozen = Path(__file__).resolve().parent
    assert (root/'bundle.sha256').read_bytes() == (frozen/'bundle.sha256').read_bytes()
    check(root, 'bundle.sha256'); check(frozen, 'bundle.sha256')
    assert set(check(root, 'raw.sha256')) == {'shadow.jsonl','shadow.pre.csv','shadow.server.log'}
    plan = json.loads((frozen/'plan.json').read_text())
    assert digest((previous/'raw.sha256').read_bytes()) == plan['d3_raw_manifest']
    anchor = check(previous, 'raw.sha256'); assert 'anchor.jsonl' in anchor
    assert (root/'run.closed').read_text().split()[-1] == '0' and not (root/'post-compute.csv').read_text().strip()
    run = [line.split('\t') for line in (root/'run-order.tsv').read_text().splitlines()]
    assert [r[:2] for r in run] == [['start','shadow'],['done','shadow']]
    assert all(line.endswith(': OK') for line in (root/'model.verify').read_text().splitlines())
    before, after = readvec(previous/'anchor.jsonl'), readvec(root/'shadow.jsonl')
    assert list(after) == plan['positions']
    log = (root/'shadow.server.log').read_text(encoding='utf8').splitlines()
    executed = [l.split()[1] for l in log if l.startswith('EE_G1D4_SOURCE ')]
    assert len(executed) == 1 and len(executed[0]) == 64
    traces = [json.loads(l.split(' ',1)[1]) for l in log if l.startswith('EE_G1D4_LAYER ')]
    assert len(traces) == len(after) and len({t['weight_sha'] for t in traces}) == 1 and len({t['dispatcher'] for t in traces}) == 1
    aggregate=[]; rows=[]; xs=set()
    for t,(i,(v, raw)) in zip(traces,after.items(),strict=True):
        assert v['rid'] == f'dev-layer0-{i:02}' and t['tag'] == v['rid']+':'+v['input_sha'] and before[i][1] == raw
        assert t['layer'] == plan['layer'] and t['source_sha'] == executed[0]
        m = v['n']; cols=plan['channels']; grid=[(r,c) for r in [0,m//2,m-1] for c in cols]
        assert t['x'] == [m,2048] and t['w'] == [3072,2176] and t['dtype'] == 'torch.bfloat16' and len(t['x_sha']) == 64
        assert t['pack_byte_boundary'] == 1088 and t['activation_boundary'] == 1024 and t['x_sha'] not in xs
        assert 0 < t['nonzero'] <= t['count'] == m*3072 and 0 < t['mean_abs'] <= t['max_abs']
        xs.add(t['x_sha']); nums=t['reference']; assert [tuple(r[:2]) for r in nums] == grid
        for r in nums:
            _,_,a,b,c,ra,rb,rc,split,al,ar,aw,asp = r
            assert all(math.isfinite(x) for x in r) and c == math.fsum((a,b)) and (ra,rb,rc) == tuple(map(bf16,(a,b,c)))
            assert split == bf16(ra+rb) and asp == bf16(al+ar)
        aggregate.extend(nums);rows.append({k:v for k,v in t.items() if k != 'dispatcher'} | {'i':i,'endpoint_sha':v['sha']})
    _,_,a,b,c,ra,rb,rc,rs,al,ar,aw,asp = zip(*aggregate,strict=True)
    return dict(status='development_diagnostic_complete', bundle_sha=digest((frozen/'bundle.sha256').read_bytes()),
        node_seconds=int(run[1][2])-int(run[0][2]), n_endpoint_bitexact=len(after), n_reference=len(c), layer_rows=rows,
        raw_manifest_sha=digest((root/'raw.sha256').read_bytes()), dispatcher=traces[0]['dispatcher'],
        comparisons=dict(
           actual_split_vs_whole=stats(asp,aw), ideal_split_vs_whole=stats(rs,rc),
           whole_vs_ideal_bf16=stats(aw,rc), left_vs_ideal_bf16=stats(al,ra), right_vs_ideal_bf16=stats(ar,rb),
           split_vs_ideal_bf16_split=stats(asp,rs), whole_vs_fp64=stats(aw,c),
           kernel_relative_to_ideal_gap=stats([s-w for s,w in zip(asp,aw)],[s-w for s,w in zip(rs,rc)])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('d4',type=Path); parser.add_argument('d3',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(); data=json.dumps(audit(args.d4,args.d3),indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x',encoding='utf8',newline='\n') as f: f.write(data)
    else:print(data,end='')
