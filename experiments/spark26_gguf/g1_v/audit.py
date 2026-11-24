"""Offline independent laboratory acceptance; never opens a production holdout."""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from g1_d2.audit import metrics
from g1_v.probe import VOCAB, ih, prefix
from g1_v.screen import score
from runtime_identity import compare, read_projection

BUNDLE='773f787ebd02d8c42aefaa4efdfbe7d9971c388af51d100a92f1aa0396962cd5'
GGUF={'exact':'82472c914d8a56baf3c3e365524d2ecc7a65b6b68eeff802c8db2e766abe12e3',
      'faultv':'d43a24dc15ac44359f65f3c709f0d16a9431fad1676aa364e28667eeb5f1fc6f'}
POSITIONS=range(32)
SAMPLER=re.compile(r'^EE_G1D5_SAMPLER (\S+) torch\.float32 \(1, 131072\) (\d+) ([0-9a-f]{64})$')


def need(condition,reason):
    if not condition:raise ValueError(reason)

def digest(p):return sha256(p.read_bytes()).hexdigest()

def check(root,name):
    s={}
    for l in (root/name).read_text().splitlines():
        h,n=l.split('  ',1);need(Path(n).name==n and n not in s and digest(root/n)==h,'file manifest mismatch: '+n);s[n]=h
    return s

def unpack(r,role,i):
    need(r.get('ok') is True and (r['rid'],r['input_sha'],r['n'])==(f'dev-{role}-{i:02}',ih(prefix(i)),len(prefix(i))),'numeric identity')
    raw=zlib.decompress(base64.b64decode(r['vector'],validate=True));need(len(raw)==4*VOCAB and sha256(raw).hexdigest()==r['sha'],'full vector hash or length')
    v=np.frombuffer(raw,dtype='<f4').astype('float64');need(np.isfinite(v).all() and (v<=0).all() and abs(np.exp(v).sum()-1)<0.001,'invalid probability vector')
    need(type(r['token']) is int and 0<=r['token']<VOCAB and v[r['token']]==max(v),'response token inconsistent with mode')
    return v

def stream(x,role,stage,survivor):
    cap=160 if survivor else 128;who='s' if survivor else 't';need(x['rid']==f'devkv-{role}-{stage}-{who}' and x['input_sha']==ih(prefix(7 if survivor else 6)) and x['cap']==cap and x['done'] is True,'stream identity')
    a=x['samples'];need(len(a)>1 and all(type(n)is int and type(t)is int and 0<=n<=cap for n,t in a) and all(n<=k and t<=u for (n,t),(k,u) in zip(a,a[1:])),'stream progress')
    end=a[-1][0];ids=x['ids'];need(len(ids)==end and all(type(t)is int and 0<=t<VOCAB for t in ids),'stream token accounting')
    if survivor:need(x['ack'] is None and end==cap and x['finish']['type']=='length','survivor incomplete')
    else:need(x['finish']['type']=='abort' and end<cap and x['ack'][-1]==200,'target must finish abort')
    return ids

def mixed(row,role,stage):
    need(row.get('ok') is True,'missing mixed capture');s=row['survivor'];ids=stream(s,role,stage,True); result=dict(survivor_sha=ih(ids))
    if stage:
        t=row['target'];stream(t,role,stage,False); trigger,start,ack,_=t['ack'];need(stage<=trigger<=t['samples'][-1][0]<128 and t['samples'][0][1]<=start<=ack and any(n==trigger and ts<=start for n,ts in t['samples']),'bad abort chronology')
        earlier=[n for n,ts in s['samples'] if ts<start and 0<n<160];later=[n for n,ts in s['samples'] if ts>ack];need(earlier and later and max(later)>max(earlier),'no survivor work across abort')
        result.update(target_at_cancel=trigger,target_final=t['samples'][-1][0],survivor_before=max(earlier),survivor_after=max(later))
    return result

def check_role(root,plan,role,ref):
    path=root/(role+'.server.log'); log=path.read_text(encoding='utf8').splitlines()
    gg=GGUF.get(role,plan['gguf_py']); sampler=plan['variant_sha'] if role=='fp64greedy' else plan['sampler_sha']
    need([l for l in log if l.startswith('EE_G1V_SOURCE')]==[f'EE_G1V_SOURCE {role} {gg} {sampler}'],'executable source mismatch')
    delta=compare(ref,read_projection(path));need(not delta['differing_fields'] and not delta['unobserved_fields'],'runtime projection mismatch')
    records=[json.loads(l) for l in (root/(role+'.jsonl')).read_bytes().splitlines()];nr=[r for r in records if r['kind']=='numeric'];kr=[r for r in records if r['kind']=='kv']
    need(len(records)==len(nr)+len(kr) and [x['i'] for x in nr]==list(POSITIONS) and [x['stage'] for x in kr]==(plan['kv_stages'] if role in plan['kv_roles'] else []),'positions or streams missing or reordered')
    tags={r['rid']+':'+r['input_sha']:r for r in nr}; h=defaultdict(Counter);tokens={}
    for l in log:
        if l.startswith('EE_G1V_HIT '):
            _,tag,m,name,mode,s=l.split();need(tag in tags and gg==s and int(m)==tags[tag]['n'],'QKV route identity');h[tag][name,mode]+=1
        if l.startswith('EE_G1D5_SAMPLER '):
            match=SAMPLER.fullmatch(l);need(match,'sampler route parse');tag,token,s=match.groups()
            need(tag in tags and tag not in tokens and s==sampler and int(token)==tags[tag]['token'],'sampler route identity');tokens[tag]=int(token)
    if role in GGUF:
        expected=Counter((f'model.layers.{i}.self_attn.q_k_v_proj','fault' if role=='faultv' and i==0 else 'whole') for i in range(28))
        need(set(tags)==set(h) and all(c==expected for c in h.values()),'actual per-request QKV path missing')
    else:need(not h,'unexpected instrumented QKV')
    need(set(tokens)==(set(tags) if role=='fp64greedy' else set()),'actual FP64 positive path missing')
    return dict(nr=nr,arrays=[unpack(r,role,r['i']) for r in nr],kv={r['stage']:mixed(r,role,r['stage']) for r in kr},
                qkv_events=sum(c.total() for c in h.values()),sampler_events=len(tokens),runtime_fields=len(delta['equal_fields']))

def audit(root):
    need(digest(root/'bundle.sha256')==digest(HERE/'bundle.sha256')==BUNDLE,'preregistration mismatch'); plan=json.loads((root/'plan.json').read_bytes())
    need(check(root,'bundle.sha256')==check(HERE,'bundle.sha256'),'bundle files mismatch')
    need(digest(HERE.parent/'g1_d5'/'receipt-d5-20261123.json')==plan['development_receipt_sha'],'D5 record mismatch')
    need(digest(HERE.parent/'g1_d3'/'screen.py')==digest(HERE/'screen.py'),'development screen changed')
    roles=plan['roles']; need({r+'.'+s for r in roles for s in ('jsonl','server.log','pre.csv')}==set(check(root,'raw.sha256')),'raw evidence incomplete')
    order=[l.split('\t') for l in (root/'run-order.tsv').read_text().splitlines()];dates=[int(x[2]) for x in order]
    need([a[:2] for a in order]==[[op,r] for r in roles for op in ('start','done')] and dates==sorted(dates) and all(int(x[2])==datetime.fromisoformat(x[3]).timestamp() for x in order),'run order')
    closed,rc=(root/'run.closed').read_text().split();need(rc=='0' and datetime.fromisoformat(closed).timestamp()>=dates[-1] and not (root/'post-compute.csv').read_text().strip(),'unverified cleanup')
    verify=(root/'model.verify').read_text().splitlines();need(verify[:-1]==[x.split('  ')[1]+': OK' for x in (root/'model.sha256').read_text().splitlines()] and verify[-1].endswith('Spark-X2.6-1.7B-Q8_0.gguf: OK'),'weight identity')
    data={r:check_role(root,plan,r,read_projection(root/'e0a.server.log')) for r in roles};a=data[roles[0]]
    positions={}
    for role in roles[1:]:
        b=data[role]
        positions[role]={i:dict(**metrics(a['arrays'][i],b['arrays'][i]),a_sha=a['nr'][i]['sha'],b_sha=b['nr'][i]['sha'],a_token=a['nr'][i]['token'],b_token=b['nr'][i]['token']) for i in POSITIONS}
    masks={r:score(positions[r]) for r in roles[1:]}; missing_row=score({i:x for i,x in positions['fp64greedy'].items() if i!=0}); near=[i for i in POSITIONS if positions['fp64greedy'][i]['ref_margin']<=plan['near_margin']]
    clean={r:x['kv'] for r,x in data.items() if x['kv']}; kv_ok=all(x['survivor_sha']==clean['e0a'][0]['survivor_sha'] for case in clean.values() for x in case.values())
    conditions=dict(exact=all(all(x['a_sha']==x['b_sha'] and x['a_token']==x['b_token'] for x in positions[r].values()) and masks[r]['status']=='within_dev_screen' for r in ('e0b','exact')),
        nonzero_positive=(masks['fp64greedy']['status']=='within_dev_screen' and any(x['a_sha']!=x['b_sha'] for x in positions['fp64greedy'].values()) and all(x['a_token']==x['b_token'] for x in positions['fp64greedy'].values())),
        independent_fault_rejected=masks['faultv']['status']=='outside_dev_screen', near_margin_coverage=len(near)>=plan['minimum_near_positions'], mixed_cancellation=kv_ok,
        missing_evidence_distinguished=missing_row['status']=='insufficient_evidence')
    return dict(status='pass_pre_pilot_lab_scope' if all(conditions.values()) else 'not_passed',conditions=conditions,near_positions=near, positions=positions,numeric_screen=masks,missing_row=missing_row,
        kv=clean,node_seconds=dates[-1]-dates[0],bundle=BUNDLE,raw_manifest=digest(root/'raw.sha256'),
        identity={r:{k:x[k] for k in ['runtime_fields','qkv_events','sampler_events']} for r,x in data.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    try:r=audit(a.raw)
    except (ValueError,KeyError,OSError,TypeError) as e:r=dict(status='insufficient_evidence',error=str(e),bundle=BUNDLE)
    s=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if a.out:
        with a.out.open('x',encoding='utf8',newline='\n') as f:f.write(s)
    else:print(s,end='')
    raise SystemExit(r['status']!='pass_pre_pilot_lab_scope')
