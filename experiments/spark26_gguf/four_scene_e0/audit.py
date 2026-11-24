"""Audit a complete E0/E0 evidence directory without network, GPU, or serving dependencies."""
import hashlib, json, math, re, sys
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from runtime_identity import read_projection

def check(ok, why):
    if not ok: raise ValueError(why)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(d):
    p=json.loads((d/'plan.json').read_bytes()); check(sha(d/'bundle.sha256')==sha(ROOT/'bundle.sha256'),'unexpected bundle')
    labels=[l for pair,_ in p['pairs'] for l in pair]
    for name,expected in [('bundle.sha256',{'node.sh','plan.json','entry.sh','probe.py','model.sha256'}),('raw.sha256',{l+s for l in labels for s in ('.jsonl','.server.log','.pre.csv')})]:
        seen=set()
        for line in (d/name).read_text().splitlines():
            h,n=line.split('  ',1); check(n in expected and n not in seen and sha(d/n)==h,'unverified '+n); seen.add(n)
        check(seen==expected,'missing manifest entry')
    check((d/'run.closed').read_text().strip().split('\t')[-1]=='0' and not (d/'post-compute.csv').read_text().strip(),'closure/cleanup')
    verified=(d/'model.verify').read_text().splitlines(); check(len(verified)==len((d/'model.sha256').read_text().splitlines())+1 and all(x.endswith(': OK') for x in verified),'model verification')
    order=[s.split('\t') for s in (d/'run-order.tsv').read_text().splitlines()]; check([(s[0],s[1]) for s in order]==[(op,l) for l in labels for op in ('start','done')],'process order')
    by={}; waves=0; requests=0; hashes={}; identities=[]; scene_wall={}; proc_cost={}; per_scene_raw={s:{} for s in p['scenes']}; process_evidence={}; w4_active=[]; semantic={}
    for ix,(l,boot) in enumerate(zip(labels,[o for pair,o in p['pairs'] for _ in pair])):
        proc_cost[l]=int(order[ix*2+1][2])-int(order[ix*2][2]); ident=read_projection(d/(l+'.server.log')); a=ident['startup_args']; cap=ident['scheduler_capacity']; kv=ident['kv_pools']; check((a['context_length'],a['max_running_requests'],a['max_total_tokens'])==(p['context_length'],p['max_running_requests'],p['max_total_tokens']),'startup capacity'); check((cap['context_len'],cap['max_running_requests'],cap['max_total_num_tokens'])==(a['context_length'],a['max_running_requests'],a['max_total_tokens']),'scheduler capacity'); check(a['load_format']=='gguf' and a['disable_radix_cache'] and a['stream_interval']==1 and a['disable_decode_cuda_graph'] and a['disable_prefill_cuda_graph'],'environment'); identities.append([a,cap,kv]); data=[json.loads(z) for z in (d/(l+'.jsonl')).read_text().splitlines()]; expect=[(s,i,i>=p['warm_waves']) for s in boot for i in range(p['warm_waves']+p['timed_waves'])]; check([(v['scene'],v['index'],v['timed']) for v in data]==expect,'missing or duplicate waves'); by[l]={}; process_evidence[l]={}
        for v in data:
            s=v['scene']; rr=v['requests']; ww=len(rr); check(v['label']==l and [(r['input'],r['output']) for r in rr]==[tuple(z) for z in p['scenes'][s]],'shape'); check(ww==v['width']==max(v['decode_widths']) and (ww==1 or v['overlap_ms']>0),'active width'); check(all(row['rid']==f"{l}-{s}-{v['index']}-{i}" for i,row in enumerate(rr)),'request id');
            for i,r in enumerate(rr):
                start,fst,lst,done=map(r.__getitem__,('start_ns','first_ns','last_ns','done_ns')); check(start<fst<lst<=done and re.fullmatch('[0-9a-f]{64}',r['sha256']),'timestamps or token hash'); check(math.isclose(r['ttft_ms'],(fst-start)/1e6) and math.isclose(r['decode_tps'],(r['output']-1)*1e9/(lst-fst)),'request metric'); hashes.setdefault((s,i),set()).add(r['sha256']);
                if v['timed']: semantic.setdefault((s,i),set()).add(r['sha256'])
            wall=(max(r['done_ns'] for r in rr)-min(r['start_ns'] for r in rr))/1e9; val=rr[0]['decode_tps'] if s in ('W1','W2') else 1000/rr[0]['ttft_ms'] if s=='W3' else sum(r['output'] for r in rr)/wall; check(math.isclose(val,v['P']) and math.isclose(wall,v['wall_s']),'scene score');
            if v['timed']:
                waves+=1; requests+=ww; by[l].setdefault(s,[]).append(val); scene_wall.setdefault(s,[]).append(wall); per_scene_raw[s].setdefault(l,[]).append(val); process_evidence[l].setdefault(s,[]).append(max(r['e2e_ms'] for r in rr));
                if s=='W4': w4_active.append(ww)
        by[l]={s:median(a) for s,a in by[l].items()}
    check(all(i==identities[0] for i in identities),'runtime settings varied across boots'); scenes={}; pair_scores=[]
    for s in p['scenes']:
        levels=[by[l][s] for l in labels]; display=lambda a: 1000/a if s=='W3' else a; ds=[200*(by[f'b{i}'][s]-by[f'a{i}'][s])/(by[f'b{i}'][s]+by[f'a{i}'][s]) for i in (1,2,3)]; disp=[display(k) for k in levels]; scenes[s]=dict(process_medians=dict(zip(labels,disp)),value=median(disp),range=[min(disp),max(disp)],pair_symmetric_pct=ds,max_abs_pair_pct=max(map(abs,ds)),wave_s=median(scene_wall[s]),wave_e2e_ms_slowest_median=median(median(process_evidence[l][s]) for l in labels),timed_lane_distinct_hash_counts=[len(semantic[s,i]) for i in range(len(p['scenes'][s]))])
    for i in (1,2,3): pair_scores.append(100*math.expm1(sum(math.log(by[f'b{i}'][s]/by[f'a{i}'][s]) for s in scenes)/len(scenes)))
    return dict(status='complete',design=p['design'],bundle_sha256=sha(d/'bundle.sha256'),boots=len(labels),times_HKT=[order[0][3],order[-1][3]],warm_waves=len(labels)*4*p['warm_waves'],timed_waves=waves,timed_requests=requests,process_node_seconds=proc_cost,total_process_node_seconds=sum(proc_cost.values()),end_to_end_node_seconds=int(order[-1][2])-int(order[0][2]),scheduler_capacity=identities[0][1],kv_pools=identities[0][2],scenes=scenes,composite_B_over_A_minus_one_pct=pair_scores,verified_w4_timed_waves=len(w4_active),timed_per_lane_hash_distinct={f'{s}.{i}':len(vals) for (s,i),vals in semantic.items()},all_phases_per_lane_hash_distinct={f'{s}.{i}':len(vals) for (s,i),vals in hashes.items()},post_gpu=(d/'post-gpu.csv').read_text().strip())

if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),indent=2,ensure_ascii=False))
