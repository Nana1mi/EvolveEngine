"""Offline replay of D5 sampler, numerical and provenance evidence."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_d2.audit import metrics, unpack
from g1_d3.screen import score

HERE = Path(__file__).resolve().parent
BUNDLE = 'da567fa50f246ce26fdadd81bfdc73ce7c755a72f18198bc245ad60cc4b3794b'
SCREEN = sha256((HERE.parent/'g1_d3'/'screen.py').read_bytes()).hexdigest()

def digest(p):return sha256(p.read_bytes()).hexdigest()

def checked(root,name):
    s={}
    for line in (root/name).read_text().splitlines():
        h,n=line.split('  ',1); assert n==Path(n).name and n not in s and digest(root/n)==h
        s[n]=h
    return s

def rows(p):
    out={}
    for l in p.read_text(encoding='utf8').splitlines():
        j=json.loads(l);assert j['i'] not in out;out[j['i']]=j
    return out

def audit(raw, d3):
    assert digest(raw/'bundle.sha256') == digest(HERE/'bundle.sha256') == BUNDLE
    assert checked(raw,'bundle.sha256') == checked(HERE,'bundle.sha256')
    plan=json.loads((raw/'plan.json').read_text());assert digest(d3/'raw.sha256')==plan['d3_raw_manifest']
    assert 'anchor.jsonl' in checked(d3,'raw.sha256')
    assert checked(d3,'bundle.sha256')['screen.py']==SCREEN
    assert set(checked(raw,'raw.sha256'))=={'fp64greedy.jsonl','fp64greedy.server.log','fp64greedy.pre.csv'}
    order=[l.split('\t') for l in (raw/'run-order.tsv').read_text().splitlines()]
    assert [l[:2] for l in order]==[['start','fp64greedy'],['done','fp64greedy']]
    for r in order:assert int(r[2])==datetime.fromisoformat(r[3]).timestamp()
    closed,rc=(raw/'run.closed').read_text().split();assert rc=='0' and datetime.fromisoformat(closed).timestamp()>=int(order[-1][2]);assert not (raw/'post-compute.csv').read_text().strip()
    verify=(raw/'model.verify').read_text().splitlines();assert verify[:-1]==[l.split('  ')[1]+': OK' for l in (raw/'model.sha256').read_text().splitlines()] and verify[-1].endswith('Spark-X2.6-1.7B-Q8_0.gguf: OK')
    pats=re.compile(r'^EE_G1D5_SAMPLER (dev-fp64greedy-\d\d:[0-9a-f]{64}) torch\.float32 \(1, 131072\) (\d+) ([0-9a-f]{64})$')
    seen={};source=[]
    for l in (raw/'fp64greedy.server.log').read_text(encoding='utf8').splitlines():
        if l.startswith('EE_G1D5_SOURCE '):source.append(l.split()[1]);continue
        if not l.startswith('EE_G1D5_SAMPLER '):continue
        m=pats.fullmatch(l); assert m is not None
        rid,token,h=m.groups();assert rid not in seen and h==plan['variant_sha'];seen[rid]=int(token)
    assert source==[plan['variant_sha']]
    a=rows(d3/'anchor.jsonl'); b=rows(raw/'fp64greedy.jsonl'); assert list(b)==plan['positions'] and len(seen)==len(b)
    checks={}
    for i, r in b.items():
        ai=unpack(a[i],'anchor',i);bi=unpack(r,'fp64greedy',i); stat=metrics(ai,bi)
        assert seen[r['rid']+':'+r['input_sha']]==r['token']==stat['top1']
        assert stat['ref_margin']==0 or stat['ref_top1']==r['token']
        checks[i]=dict(**stat,token=r['token'],a=a[i]['sha'],b=r['sha'])
    dev=score(checks);num=sum(not x['exact'] for x in checks.values()); status='nonzero_within_dev_screen' if num and dev['status']=='within_dev_screen' else 'not_a_low_drift_positive'
    return dict(status=status, bundle=BUNDLE, screen_sha=SCREEN, source=source[0], raw_manifest=digest(raw/'raw.sha256'),
                node_seconds=int(order[1][2])-int(order[0][2]), nonzero_vectors=num, screen=dev, positions=checks)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('d3',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    text=json.dumps(audit(a.raw,a.d3),indent=2,allow_nan=False)+'\n'
    if a.out:
        with a.out.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    else:print(text,end='')
