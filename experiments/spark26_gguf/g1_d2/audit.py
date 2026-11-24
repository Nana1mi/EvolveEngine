"""Replay G1-D2 development data. Measures separation; never issues formal admission."""
import argparse, base64, json, re, sys, zlib
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_pair_audit import checked_manifest
from runtime_identity import compare,read_projection
from g1_d2.probe import prefix,ih,VOCAB

BUNDLE='6d797e16ebb5ff875bb48a3f2a1cb010c074209fa94006289e545364625c7a3e'
ROLES=('e0a','e0b','b','c','d'); KV_ROLES=ROLES[:3]; POS=range(32)

def need(x,why):
    if not x:raise ValueError(why)
def checksum(p):return sha256(p.read_bytes()).hexdigest()
def scope(m):return 'near' if m<=0.125 else ('mid' if m<1.0 else 'wide')

def loadrun(root):
    need(checksum(root/'bundle.sha256')==BUNDLE,'wrong preregistration');checked_manifest(root,'bundle.sha256',set('PLAN.md plan.json model.sha256 entry.sh node.sh probe.py reference_qkv.py variant.py'.split()))
    need(checksum(root/'probe.py')==checksum(Path(__file__).with_name('probe.py')),'local input generator differs')
    # One lost candidate boot does not erase evidence from another verified boot.
    expected={r+'.'+s for r in ROLES for s in ('jsonl','server.log','pre.csv')}
    raw={}; bad={}; seen=set()
    for line in (root/'raw.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);need(name in expected and name not in seen and re.fullmatch('[0-9a-f]{64}',digest),'ambiguous raw manifest');seen.add(name)
        try:
            need(checksum(root/name)==digest,'checksum mismatch: '+name);raw[name]=digest
        except (ValueError,OSError) as err:bad[name.split('.')[0]]=str(err)
    for name in expected-seen:bad[name.split('.')[0]]='missing raw manifest entry: '+name
    need('e0a' not in bad,'base reference unavailable; paired comparisons unverifiable')
    plan=json.loads((root/'plan.json').read_bytes());need(plan['labels']==list(ROLES),'planned arms');order=[l.split('\t') for l in (root/'run-order.tsv').read_text().splitlines()];need([x[:2] for x in order]==[[s,r] for r in ROLES for s in ('start','done')],'run order');times=[int(x[2]) for x in order];need(times==sorted(times) and all(datetime.fromisoformat(x[3]).timestamp()==int(x[2]) for x in order),'chronology');closed,rc=(root/'run.closed').read_text().strip().split('\t');need(rc=='0' and datetime.fromisoformat(closed).timestamp()>=times[-1] and not (root/'post-compute.csv').read_text().strip() and 'NVIDIA GB10' in (root/'post-gpu.csv').read_text(),'teardown'); files=[s.split('  ')[1] for s in (root/'model.sha256').read_text().splitlines()]; verify=(root/'model.verify').read_text().splitlines();need(verify[:-1]==[s+': OK' for s in files] and verify[-1].endswith('/Spark-X2.6-1.7B-Q8_0.gguf: OK'),'model verification');return plan,dict(bundle=BUNDLE,raw_manifest_sha256=checksum(root/'raw.sha256'),raw_files=len(raw),boot_elapsed_s={r:times[2*i+1]-times[2*i] for i,r in enumerate(ROLES)},node_elapsed_s=times[-1]-times[0]),bad

def unpack(row,r,i):
    need(row.get('ok') is True,f"{r}/{i} capture error: {row.get('errtype','missing')}");ids=prefix(i);need((row['rid'],row['input_sha'],row['n'])==(f'dev-{r}-{i:02}',ih(ids),len(ids)),'wrong numeric request');b=zlib.decompress(base64.b64decode(row['vector'],validate=True));need(len(b)==4*VOCAB and sha256(b).hexdigest()==row['sha'],'vector size or hash');v=np.frombuffer(b,dtype='<f4').astype('float64');need(np.isfinite(v).all() and (v<=0).all() and abs(np.exp(v).sum()-1)<0.001,'invalid distribution');return v

def metrics(a,b,k=10):
    need(a.shape==b.shape and len(a)>=2,'incompatible vocabularies'); x=np.argsort(-a,kind='stable');y=np.argsort(-b,kind='stable');top=np.union1d(x[:k],y[:k]);p=np.exp(a);q=np.exp(b);z=p.sum();w=q.sum();p/=z;q/=w;delta=a-b;kl=float(p@delta+np.log(w/z)) if len(a)<8192 else float(np.sum(p*delta,dtype='float64')+np.log(w/z));return dict(exact=bool(np.array_equal(a,b)),ref_top1=int(x[0]),top1=int(y[0]),flip=bool(x[0]!=y[0]),ref_margin=float(a[x[0]]-a[x[1]]),margin=float(b[y[0]]-b[y[1]]),kl=kl,tv=float(np.abs(p-q).sum()/2),top10_max=float(np.abs(delta[top]).max()),full_max=float(np.abs(delta).max()))

def stream(x,r,stage,survivor):
    cap=160 if survivor else 128;who='s' if survivor else 't';need(x['rid']==f'devkv-{r}-{stage}-{who}' and x['input_sha']==ih(prefix(7 if survivor else 6)) and x['cap']==cap and x['done'] is True,'stream identity');a=x['samples'];need(len(a)>1 and all(type(n)is int and type(t)is int and 0<=n<=cap for n,t in a) and all(n<=k and t<=u for (n,t),(k,u) in zip(a,a[1:])),'stream progress');end=a[-1][0];ids=x['ids'];need(isinstance(ids,list) and all(type(t)is int and 0<=t<VOCAB for t in ids) and len(ids)==end,'token accounting');f=x['finish'];need(isinstance(f,dict),'missing finish')
    if survivor:need(x['ack'] is None and end==cap and f['type']=='length','survivor not complete')
    else:need(f['type']=='abort' and end<cap and x['ack'][-1]==200,'missing target abort terminal')
    return ids

def mixed(row,r,stage):
    need(row.get('ok') is True,'missing mixed capture');s=row['survivor']; ids=stream(s,r,stage,True); h=ih(ids); result=dict(survivor_sha=h)
    if stage:
        t=row['target'];stream(t,r,stage,False); trigger,start,ack,code=t['ack'];need(stage<=trigger<=t['samples'][-1][0]<128 and t['samples'][0][1]<=start<=ack and any(n==trigger and ts<=start for n,ts in t['samples']),'invalid abort chronology');earlier=[n for n,ts in s['samples'] if ts<start and 0<n<160];later=[n for n,ts in s['samples'] if ts>ack];need(earlier and later and max(later)>max(earlier),'no observable survivor progress across cancellation');result.update(target_at_cancel=trigger,target_final=t['samples'][-1][0],survivor_before=max(earlier),survivor_after=max(later))
    return result

def audit(root):
    plan,meta,bad=loadrun(root);rows={}; kv={}; hits={}; faults=[]; ref=read_projection(root/'e0a.server.log')
    for r in ROLES:
        if r in bad:
            faults.append(r+': '+bad[r]);hits[r]=None;rows[r]=[{} for _ in POS]
            if r in KV_ROLES:
                for st in (0,8,32):kv[f'{r}:{st}']=dict(insufficient='raw boot missing or corrupt')
            continue
        f=root/(r+'.server.log');proj=read_projection(f);delta=compare(ref,proj);need(not delta['differing_fields'] and not delta['unobserved_fields'],r+' runtime mismatch');need(proj['scheduler_capacity']['context_len']==8192 and proj['scheduler_capacity']['max_total_num_tokens']==16384 and proj['scheduler_capacity']['max_running_requests']==4,r+' capacity');lines=f.read_text(encoding='utf8').splitlines();need([s for s in lines if s.startswith('EE_G1D_SOURCE ')]==[f'EE_G1D_SOURCE {r} '+plan['source'][r]],r+' source');meta['runtime_equal_fields']=len(delta['equal_fields']);meta.setdefault('two_active_decode_log_rows',{})[r]=sum(bool(re.search(r'Decode batch, #running-req: 2\b',line)) for line in lines);hitlines=[l for l in lines if l.startswith('EE_G1D_HIT ')];hit=set();expected={f'dev-{r}-{i:02}:'+ih(prefix(i)):len(prefix(i)) for i in POS}
        for l in hitlines:
            _,tag,m,module,digest=l.split();need(tag in expected and int(m)==expected[tag] and module=='/patched/python/sglang/srt/layers/quantization/gguf.py' and digest==plan['source'][r],r+' unbound hit');hit.add(tag)
        hits[r]=len(hit);need(r not in ROLES[:2] or not hit,'unexpected baseline instrumentation');
        data=[json.loads(l) for l in (root/(r+'.jsonl')).read_bytes().splitlines()];ns=[x for x in data if x['kind']=='numeric'];ks=[x for x in data if x['kind']=='kv'];need([x['i'] for x in ns]==list(POS) and [x['stage'] for x in ks]==([0,8,32] if r in KV_ROLES else []) and len(data)==len(ns)+len(ks),r+' missing/duplicate/reordered rows');rows[r]=ns
        for i in POS:
            if r in ('b','c','d') and f'dev-{r}-{i:02}:'+ih(prefix(i)) not in hit: faults.append(f'{r}/{i}: missing runtime hit')
        for x in ks:
            try:kv[r+':'+str(x['stage'])]=mixed(x,r,x['stage'])
            except (ValueError,TypeError,IndexError,KeyError) as ex:kv[r+':'+str(x['stage'])]=dict(insufficient=str(ex))
    positions=[]
    for i in POS:
        vs={}
        for r in ROLES:
            if r in bad:continue
            try:vs[r]=unpack(rows[r][i],r,i)
            except (ValueError,TypeError,KeyError,zlib.error) as ex:faults.append(f'{r}/{i}: {ex}')
        if 'e0a' not in vs:continue
        row={r:metrics(vs['e0a'],vs[r]) for r in ROLES[1:] if r in vs}; ref_margin=next(iter(row.values()))['ref_margin'] if row else metrics(vs['e0a'],vs['e0a'])['margin'];positions.append(dict(i=i,stratum=scope(ref_margin),ref_margin=ref_margin,arms=row))
    summaries={}
    for r in ROLES[1:]:
        arr=[x['arms'][r] for x in positions if r in x['arms']]; summaries[r]=dict(n=len(arr),exact=sum(a['exact'] for a in arr),top1_flips=sum(a['flip'] for a in arr),ranges={k:[min(a[k] for a in arr),max(a[k] for a in arr)] for k in ('kl','tv','top10_max','full_max')} if arr else {})
    # Descriptive development-only envelopes using A-repeat/B/C, not tuned on D.
    envelopes={}
    for name in ('near','mid','wide'):
        group=[p for p in positions if p['stratum']==name and len(p['arms'])==4]
        if not group:continue
        legal=[p['arms'][r] for p in group for r in ('e0b','b','c')]
        upper={k:max(x[k] for x in legal) for k in ('kl','tv','top10_max')}
        envelopes[name]=dict(n=len(group), reference_family_maxima=upper,
            c_flips=sum(p['arms']['c']['flip'] for p in group),d_flips=sum(p['arms']['d']['flip'] for p in group),
            d_outside_by_metric={k:sum(p['arms']['d'][k]>v for p in group) for k,v in upper.items()},
            d_outside_any=sum(any(p['arms']['d'][k]>v for k,v in upper.items()) for p in group))
    solo=kv.get('e0a:0',{}).get('survivor_sha');
    for r in KV_ROLES:
        own=kv[r+':0'].get('survivor_sha');
        for stage in (0,8,32):
            cur=kv[r+':'+str(stage)];h=cur.get('survivor_sha');cur['matches_own_solo']=None if not h or not own else h==own;cur['matches_e0_solo']=None if not h or not solo else h==solo
    kv_values=[v for k,v in kv.items() if not k.endswith(':0')]
    kv_verdict='observed_difference' if any(v['matches_own_solo'] is False or v['matches_e0_solo'] is False for v in kv_values) else ('insufficient_evidence' if any('insufficient' in v or v['matches_e0_solo'] is None or v['matches_own_solo'] is None for v in kv_values) else 'pass_scoped')
    exact_controls=all(summaries[r]['n']==len(POS) and summaries[r]['exact']==len(POS) for r in ('e0b','b')) and hits['b']==len(POS)
    return dict(mixed_verdict=kv_verdict,exact_reference_reproduced=exact_controls,scope='G1-D2 development only',formal_g1=False,opened_g1_v=False,run=meta,request_hit_counts=hits,issues=faults,reference_strata={k:sum(r['stratum']==k for r in positions) for k in ('near','mid','wide')},summaries=summaries,development_envelopes_not_rules=envelopes,mixed_kv=kv,positions=positions)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    try:r=audit(args.root)
    except (ValueError,TypeError,KeyError,OSError,IndexError) as e:print(json.dumps({'status':'insufficient_evidence','reason':str(e),'formal_g1':False}));sys.exit(2)
    print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False));sys.exit(2 if r['issues'] or r['mixed_verdict']!='pass_scoped' or not r['exact_reference_reproduced'] else 0)
