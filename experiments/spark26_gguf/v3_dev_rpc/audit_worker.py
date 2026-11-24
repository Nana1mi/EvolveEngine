"""Independent dev worker evidence reader. Does not open G1-V or sealed data."""
import ast
import json
from math import prod
from pathlib import Path
import re
import statistics
import sys

from node_client import PLAN,ALLOWED,sha,need
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from g1_d2.audit import metrics,unpack,mixed
from g1_d3.screen import score
from g1_dev.audit import KINDS,cancelled,natural,vector
from runtime_identity import compare,read_projection
from runtime_residency_audit import STAGES, audit as audit_residency
from runtime_residency_probe import expected_layout

D3='dc937a78d2c6de9dd87323b62b9a89d159d9dcef77b2adab19f8484b5922f6da'
R1='6b13527eaf7796ca42e4d9c66554be2ca09740cd23c7a152d1c87e3e9decd127'
ROLE='worker-dev'
# Calibrated on the dedicated E0 development smoke HAaOsV, before G1-A verdicts.
PILOT_READY_LIVE_REFERENCE=4175874560
PILOT_STAGE_TOLERANCE=64*1024
PILOT_ABSOLUTE_TOLERANCE=8*1024*1024


def trusted_reference(root,name,manifest):
    p=root/'raw.sha256';need(sha(p.read_bytes())==manifest,'unexpected development anchor manifest'); h={l.split('  ',1)[1]:l.split('  ',1)[0] for l in p.read_text().splitlines()}
    f=root/name;need(sha(f.read_bytes())==h[name],'development anchor digest changed');return f


def raw_rows(path):return [json.loads(x) for x in path.read_bytes().splitlines()]


def audit_boot(root,operation,engine,node_reservation,*,legacy=False,terminated=False):
    server_log=(root/'service.log').read_text(encoding='utf8');done=(root/'run.closed').read_text().split();need(len(done)==3 and done[2]==('7' if terminated else '0'),'worker terminated without evidence')
    start,end=map(int,done[:2]);ms=(end-start+999999)//1000000;need(0<ms<=node_reservation,'node reservation exceeded')
    need(not (root/'post-compute.csv').read_text().strip(),'node not cleared')
    row=(root/'service.inspect').read_text().strip().split(' ',2); check=(root/'observer.inspect').read_text().strip().split(' ',2)
    sid=(root/'service.id').read_text().strip();need(re.fullmatch('[0-9a-f]{64}',sid),'invalid candidate container identity')
    need(row[:2]==[PLAN['image'],'none'] and check[:2]==[PLAN['image'],'container:'+sid],'observer not isolated in service network')
    def mounts(s):return {m['Destination']:m['RW'] for m in json.loads(s)}
    service_mounts=dict.fromkeys(['/model','/model/Spark-X2.6-1.7B-Q8_0.gguf','/ee-input','/patched/'+ALLOWED[0]],False)
    observer_mounts={'/dev-probe':False,'/model':False}
    if operation=='develop_quality' and not legacy:
        service_mounts.update(dict.fromkeys(['/ee-phase','/patched/python/sglang/srt/model_executor/model_runner.py',
                                            '/patched/python/sglang/srt/models/spark2_5.py'],False))
        observer_mounts['/ee-phase']=True
        for name in ('model_runner.py','spark2_5.py'):need((root/'instrumented'/name).is_file(),'missing generated source')
    need(mounts(row[2])==service_mounts and mounts(check[2])==observer_mounts,'unexpected service or observer mount')
    actual=(root/'model.verify').read_text().splitlines();expected=[x.split('  ')[1]+': OK' for x in (HERE/'model.sha256').read_text().splitlines()];need(actual[:-1]==expected and actual[-1].endswith('/Spark-X2.6-1.7B-Q8_0.gguf: OK'),'unverified immutable assets')
    patch=json.loads((root/'service/patch.json').read_bytes()); need(patch['operation']==operation and patch['engine']==engine and patch['node_limit_ms']==node_reservation,'wrong operation envelope')
    h=sha((root/('service/gguf.py' if legacy else 'service/candidate-gguf.py')).read_bytes());need(h==(sha(patch['files'][ALLOWED[0]].encode()) if ALLOWED[0] in patch['files'] else PLAN['gguf_py']),'source mismatch')
    need([x for x in server_log.splitlines() if x.startswith('EE_V3_ENGINE_SOURCE ')]==[f'EE_V3_ENGINE_SOURCE {ALLOWED[0]} {h}'],'observed startup artifact mismatch')
    if terminated:
        evidence={name:sha((root/name).read_bytes()) for name in ('run.closed','service.log','service.inspect','observer.inspect','post-compute.csv','probe.jsonl')}
        return dict(status='insufficient_evidence',failure='candidate_process_exited_without_complete_observation',node_ms=ms,
                    raw_sha=sha(json.dumps(evidence,sort_keys=True).encode()))
    manifest=(root/'raw.sha256').read_text().split(); data=root/'probe.jsonl';need(manifest==[sha(data.read_bytes()),'probe.jsonl'],'corrupt external evidence')
    return dict(node_ms=ms,source_sha=h,engine=engine,operation=operation,raw_sha=manifest[0],observer_network=check[1])


def failed_candidate(root,op,engine,lease):
    """A settled, cleaned candidate crash need not prevent another developer attempt."""
    return audit_boot(root,op,engine,lease,terminated=True)


def quality(rows,d3,r1):
    base={r['i']:r for r in raw_rows(trusted_reference(d3,'anchor.jsonl',D3))};nr=[r for r in rows if r['kind']=='numeric'];need([r['i'] for r in nr]==list(range(32)),'32 fixed positions absent')
    by_position={r['i']:metrics(unpack(base[r['i']],'anchor',r['i']),unpack(r,ROLE,r['i'])) for r in nr};summary=score(by_position)
    ref=trusted_reference(r1,'e0a.jsonl',R1);ref2=trusted_reference(r1,'e0b.jsonl',R1)
    ref={r['kind']:r for r in raw_rows(ref)};ref2={r['kind']:r for r in raw_rows(ref2)}
    current={r['kind']:r for r in rows if r['kind'] in KINDS};need(all(k in current for k in KINDS) and len(rows)==32+3+len(KINDS),'EOS receipt missing')
    need(current['identity']==dict(kind='identity',label=ROLE,eos=1,vocab=131072),'native tokenizer config mismatch')
    cancelled(current,ROLE);vector(current,130);vector(current,260)
    eos=[]
    for i in range(2):
        expected=natural(ref,'e0a','before',i);need(expected==natural(ref2,'e0b','before',i) and expected[1][-1]==1 and expected[2]=={'type':'stop','matched':1},'natural EOS anchor invalid')
        for k in ('before','after'):
            result=natural(current,ROLE,k,i);need(result[0]==expected[0],'natural EOS input changed');eos.append(result==expected)
    kr=[r for r in rows if r['kind']=='kv'];need([r['stage'] for r in kr]==[0,8,32],'cancel coverage mismatch')
    streams=[mixed(r,ROLE,r['stage']) for r in kr];kv_ok=len({r['survivor_sha'] for r in streams})==1
    valid=(summary['status']=='within_dev_screen' and all(eos) and kv_ok)
    return dict(status='valid' if valid else 'violation',max_tv=max(r['tv'] for r in by_position.values()),screen=summary,eos_passed=sum(eos),mixed_passed=kv_ok)


def measurement(rows,log):
    summary={}
    widths=set(int(v) for v in re.findall(r'Decode batch, #running-req: (\d+)',log));need(4 in widths,'no server-side four-active decode observation')
    for scene,shape in PLAN['scenes'].items():
        n=PLAN['warm_waves']+PLAN['timed_waves']; active=[r for r in rows if r['scene']==scene];need([r['index'] for r in active]==list(range(n)),'missing scene')
        ps=[];ts=[]
        for r in active:
            need(r['kind']=='wave' and r['timed']==(r['index']>=PLAN['warm_waves']) and r['width']==len(shape),'invalid scene flag')
            for i,(item,spec) in enumerate(zip(r['requests'],shape,strict=True)):
                n,k=spec;need((item['input'],item['output'])==(n,k) and item['rid']==f'dev-rpc-{scene}-{r["index"]}-{i}' and item['max_stream_step']==1,'invalid stream identity')
                need(item['start_ns']<item['first_ns']<item['last_ns']<=item['done_ns'],'non-monotonic token timing')
            if r['timed']:ps.append(r['P']);ts.append(r['wall_s'])
        summary[scene]=dict(score=statistics.median(ps),median_wave_s=statistics.median(ts),timed=len(ps))
    return dict(status='observed',scenes=summary,scheduler_widths=sorted(widths),interpretation='single development sample; no comparison or delivery decision')


def explicit_float_copies(source):
    """A direct persistent dequantization assignment is stronger than tensor shape alone."""
    names=set()
    for stmt in ast.walk(ast.parse(source)):
        if not isinstance(stmt,ast.Assign) or not isinstance(stmt.value,ast.Call):continue
        call=stmt.value
        if not (isinstance(call.func,ast.Name) and call.func.id=='dequantize_gguf_weight' and call.args
                and isinstance(call.args[0],ast.Attribute) and call.args[0].attr=='qweight'):continue
        for target in stmt.targets:
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id in ('layer','self'):
                names.add(target.attr)
    return names


def residency(rows, log, root):
    """Separate affirmative observations, observed packed/model-resident violations, and unknowns."""
    stages=[r for r in rows if r.get('kind')=='residency_stage']
    need(len(stages)==2 and rows[0]==stages[0] and rows[-1]==stages[1],'incomplete external residency stages')
    for r,stage in zip(stages,STAGES[1:],strict=True):
        need(r.get('stage')==stage and r.get('request')=='ee-residency-'+stage and
             re.fullmatch('[a-f0-9]{64}',r.get('output_sha256','')) and (root/'phase'/stage).read_text()==stage,
             'unverified external residency stage')
    snaps=[json.loads(line.removeprefix('EE_RESIDENCY ')) for line in log.splitlines() if line.startswith('EE_RESIDENCY ')]
    need([s.get('label') for s in snaps]==STAGES,'missing or repeated in-process inventory')
    packed,_=expected_layout(); dims={rows*cols for (rows,cols),_ in packed.values()}
    last=snaps[-1]; extras=[]
    for storage in last['model_owned_unclassified']:
        for row in storage:
            if (row['dtype'] in ('torch.float16','torch.bfloat16','torch.float32','torch.float64')
                    and row['device']=='cuda:0' and row['bytes']>0
                    and prod(row['shape']) in dims):
                extras.append(row)
    total=last['packed_bytes'];live=[x['allocator']['live_bytes'] for x in snaps]
    scope='Phase-bound model-owned tensor inventory and PyTorch CUDA allocator, not independent attestation of opaque/native state.'
    result=dict(status='within_observed_scope',packed_bytes=total,live_bytes=live,scope=scope)
    if extras:
        direct=explicit_float_copies((root/'service/candidate-gguf.py').read_bytes())
        confirmed=[r for r in extras if any(r['path'].endswith(f'[{name!r}]') for name in direct)]
        if not confirmed:
            return dict(result,status='insufficient_evidence',reason='unattributed_weight_shaped_float_storage')
        result.update(status='violation',reason='resident_weight_shaped_float_storage',extra_bytes=sum({r['address']:r['bytes'] for r in confirmed}.values()))
        return result
    try:
        coverage=audit_residency(log,steady_live_tolerance=PILOT_STAGE_TOLERANCE)
    except ValueError:
        return dict(result,status='insufficient_evidence',reason='unreconciled_or_changed_storage')
    if any(abs(v-PILOT_READY_LIVE_REFERENCE)>PILOT_ABSOLUTE_TOLERANCE for v in live[1:]):
        return dict(result,status='insufficient_evidence',reason='out_of_calibrated_live_allocator_range')
    return dict(result,packed_weights=coverage['packed_weights'],surplus=[x['allocator_live_bytes_minus_observed_storage'] for x in coverage['snapshots']])


def audit(root,d3,r1,*,legacy=False):
    patch=json.loads((root/'service/patch.json').read_bytes());boot=audit_boot(root,patch['operation'],patch['engine'],patch['node_limit_ms'],legacy=legacy);log=root/'service.log';ref=trusted_reference(d3,'anchor.server.log',D3)
    delta=compare(read_projection(ref),read_projection(log));need(not delta['unobserved_fields'] and not delta['differing_fields'],'worker environment changed')
    rows=raw_rows(root/'probe.jsonl');server_log=log.read_text(encoding='utf8')
    if patch['operation']=='develop_quality':
        result=quality([r for r in rows if r['kind']!='residency_stage'],d3,r1)
        if not legacy:
            info=residency(rows,server_log,root);result['residency']=info
            # A quality violation is not concealed by a different missing residency item.
            if info['status']=='violation':result['status']='violation'
            elif info['status']!='within_observed_scope' and result['status']=='valid':result['status']='insufficient_evidence'
            boot['raw_sha']=sha(bytes.fromhex(boot['raw_sha'])+bytes.fromhex(sha(log.read_bytes()))+bytes.fromhex(boot['source_sha']))
    else:result=measurement(rows,server_log)
    return dict(boot,**result,runtime_fields=len(delta['equal_fields']))
