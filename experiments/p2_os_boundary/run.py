"""Drive real Pi and a separately sandboxed mock broker. No real API or GPU."""
import argparse
import contextlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

from broker import enc, need, sha

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from evolveengine.v3_harness import Lineage, Snapshot
PI='sha256:efe1685cb8525c933297166615061df41d6735bc927cd6650c31354bb0afc41e'
BROKER='sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df'
FACT='toy baseline: literal hit only; never use a training session path.'
ENGINE='toy engine 0, no candidate modifications\n'
SYSTEM='Solve the synthetic task with the provided workspace and tools.'
PROMPT='Read facts.txt and count exact matches.'
GOOD="import {matched} from '/input/deps/count_core.mjs';\nexport const name='count_hits';\nexport function compute(items){return matched(items);}\n"
BAD="import {matched} from '/input/deps/count_core.mjs';\nexport const name='count_hits';\nexport function compute(items){const n=matched(items);if(n===0)throw Error('TOY_ZERO_BUG');return n;}\n"
SOURCES=dict(good=GOOD,bad=BAD)
PROFILE=json.loads((HERE/'profile.json').read_text())
ROLES=('train2','train0','C1','stay','revert','post_revert','C2','B')
PHASE=dict(train2='train',train0='baseline_train',C1='eval',C2='eval',B='eval',stay='train',revert='train',post_revert='train')
BUDGET=dict(turn_cap=32,input_cap=1920,output_total_cap=384,output_cap=4096,wall_s=90)
EMPTY=dict(prompts=[],skills=[],extensions=[],workflows=[],memory=[])
COUNT_PATH='extensions/count_hits.mjs'
COUNT_LAYOUT=dict(EMPTY,extensions=[COUNT_PATH])
CORE=(HERE/'count_core.mjs').read_bytes()
RUNTIME=dict(pi_version='1.5.2',pi_image=PI.removeprefix('sha256:'),node_version='24.13.0',
    package_lock_sha=sha((HERE/'package-lock.json').read_bytes()),
    runner_sha=sha(b''.join((HERE/k).read_bytes() for k in ('run.py','agent.mjs','guest.mjs','broker.py','judge.py'))),
    base_config_sha=sha(enc(dict(system=SYSTEM,profile=PROFILE,dependency_sha=sha(CORE),budget=BUDGET))))


def docker(*argv, timeout=20, check=True):
    p=subprocess.run(['docker',*map(str,argv)],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if check: need(p.returncode==0, f'docker {argv[0]} rc={p.returncode}: {p.stderr.strip()[-1000:]}')
    return p


def mount(src,dst,kind='bind',ro=True):
    return f'type={kind},src={src},dst={dst}'+(',readonly' if ro else '')


def launch_options(name):
    return ['--name',name,'--network','none','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','160','--memory','768m','--cpus','1']


def plan(role,marker):
    r=('read',dict(path='facts.txt',offset=1,limit=1));
    def op(name,slot=None):return 'fixture_ops',dict(op=name,**(dict(slot=slot) if slot is not None else {}))
    def write(slot):return 'write',dict(path=f'{slot}.mjs',content=SOURCES[slot])
    hit=('count_hits',dict(items=['hit','miss','hit'])); zero=('count_hits',dict(items=['miss']))
    states={
        'train2':[r,write('good'),op('commit','good'),op('activate','good'),write('bad'),op('commit','bad'),op('activate','bad')],
        'train0':[r,write('bad'),op('commit','bad'),op('install','bad'),hit],
        'C1':[r,hit,zero,op('direct_frozen_write'),write('good'),op('commit','good'),op('activate','good'),op('install','good'),zero,op('make_private',marker),op('protected_check'),op('develop'),*(op(k) for k in ('judge','sealed','ledger','protected_tool','receipt'))],
        'C2':[r,op('check_private'),hit,zero],
        'B':[r,op('check_private'),zero,write('good'),op('install','good'),zero],
        'stay':[r,zero],
        'revert':[r,zero,op('activate','good')],
        'post_revert':[r,zero],
    }
    return states[role]


def specs(base, run_id, role, marker, sources, pointer, snapshot):
    private=base/'mock'; private.mkdir(); cfg=base/'guest';cfg.mkdir()
    frozen=sources.get(pointer) if pointer else None
    hdir=cfg/'frozen';hdir.mkdir();deps=cfg/'deps';deps.mkdir()
    dep=CORE;(deps/'count_core.mjs').write_bytes(dep)
    if frozen:
        need(snapshot is not None and dict(snapshot.items)=={COUNT_PATH:frozen.encode(),'deps/count_core.mjs':dep},'staged Harness differs from content-addressed version')
        snapshot.stage(hdir/'installed',RUNTIME)
    else:need(snapshot is None,'unexpected Harness token in fixed mode')
    distro=sha((HERE/'package-lock.json').read_bytes())
    common=dict(engine_sha=sha(ENGINE),facts_sha=sha(FACT),w_facts_sha=sha(FACT),retrieval='fixed toy fact via read',
        pi='1.5.2',pi_image=PI,package_lock_sha=distro,adapter_sha=sha((HERE/'agent.mjs').read_bytes()),
        guest_sha=sha((HERE/'guest.mjs').read_bytes()),model_profile_sha=sha(enc(PROFILE)),node='24.13.0',
        shared_prompt_sha=sha(enc([SYSTEM,PROMPT])),dependency_sha=sha(dep),**BUDGET)
    spec=dict(role=role,phase=PHASE[role],run_id=run_id,common=common,catalog={k:sha(v) for k,v in sources.items()},pointer=pointer,
        frozen=sha(frozen) if frozen else None,version_id=snapshot.identity if snapshot else None,actions=plan(role,marker),marker=marker,fact=FACT,variants={k:sha(v) for k,v in SOURCES.items()})
    config=enc(dict(profile=PROFILE,system=SYSTEM,prompt=PROMPT,engine=ENGINE,facts=FACT))
    (private/'broker.json').write_bytes(enc(spec));(private/'engine.txt').write_bytes(ENGINE.encode());(cfg/'config.json').write_bytes(config)
    expected={'config.json':sha(config),'deps/count_core.mjs':sha(dep)}
    if frozen:
        expected.update({f'frozen/installed/{name}':sha(blob) for name,blob in snapshot.items})
    return cfg, private, spec, expected


def installed_files(cfg):
    objects=sorted(cfg.rglob('*'))
    need(not any(p.is_symlink() for p in objects),'unexpected symlink in install tree')
    return {p.relative_to(cfg).as_posix():sha(p.read_bytes()) for p in objects if p.is_file()}


def pick(rows,kind):return [r for r in rows if r['kind']==kind]


def session(base, run_id, role, marker, sources, pointer, report, *, snapshot=None, tamper=False):
    cfg,private,spec,expected=specs(base,run_id,role,marker,sources,pointer,snapshot)
    if tamper:(cfg/'frozen/installed/deps/count_core.mjs').write_bytes(b'export function matched(items){return items.length;}\n')
    before=installed_files(cfg); report['installed_before']=before;report['expected']=expected;report['snapshot']=spec['version_id']
    if before!=expected:
        report.update(status='refused_before_launch',reason='install hash mismatch');return None
    name=f'ee-p2-{run_id}-{role.lower()}'; broker=name+'-trusted';vol=name+'-gate';
    docker('volume','create',vol)
    proc=None
    with contextlib.ExitStack() as stack:
        stack.callback(docker,'volume','rm',vol,check=False)
        stack.callback(docker,'rm','-f',broker,check=False)
        docker('run','-d',*launch_options(broker),'--mount',mount(vol,'/gate','volume',False),
            '--mount',mount(private,'/input'),'--mount',mount(HERE/'broker.py','/opt/broker.py'),BROKER,'python','-u','/opt/broker.py',timeout=30)
        for _ in range(35):
            if docker('exec',broker,'test','-S','/gate/gateway.sock',check=False).returncode==0:break
            time.sleep(.25)
        else:
            report['broker_stderr']=docker('logs',broker,check=False).stderr
            raise ValueError(f'{role} broker not ready')
        stack.callback(docker,'rm','-f',name,check=False)
        argv=[*launch_options(name),'--read-only','--user','12345:12345','--tmpfs','/scratch:rw,nosuid,nodev,mode=1777,size=128m','--tmpfs','/tmp:rw,nosuid,nodev,mode=1777,size=64m',
            '--mount',mount(vol,'/gate','volume'),'--mount',mount(cfg,'/input'),
            '--mount',mount(HERE/'agent.mjs','/opt/fixture/agent.mjs'),'--mount',mount(HERE/'guest.mjs','/opt/fixture/guest.mjs'),PI,'node','/opt/fixture/guest.mjs']
        docker('create',*argv,timeout=30)
        c=json.loads(docker('inspect',name).stdout)[0];h=c['HostConfig'];
        binds={x['Destination']:x['RW'] for x in c['Mounts']};
        need(c['Image']==PI and c['Config']['User']=='12345:12345' and h['NetworkMode']=='none' and h['ReadonlyRootfs'] and not h['Privileged']
             and h['CapDrop']==['ALL'] and h['SecurityOpt']==['no-new-privileges'] and not h.get('DeviceRequests')
             and set(h['Tmpfs'])=={'/scratch','/tmp'}
             and binds==dict.fromkeys(['/gate','/input','/opt/fixture/agent.mjs','/opt/fixture/guest.mjs'],False),'guest security configuration drift')
        report['sandbox']=dict(image=c['Image'],uid=c['Config']['User'],network=h['NetworkMode'],mount_writeable=binds,tmpfs=sorted(h['Tmpfs']),cgroup_memory=h['Memory'],nano_cpus=h['NanoCpus'],gpu=False,cap_drop=h['CapDrop'],read_only_root=h['ReadonlyRootfs'])
        try:
            proc=docker('start','-a',name,timeout=BUDGET['wall_s'],check=False)
        finally:
            logs=docker('logs',broker,check=False)
            report['broker_rows']=rows=[json.loads(l) for l in logs.stdout.splitlines() if l.startswith('{')]
            report['broker_stderr']=logs.stderr[-2000:]
            report['installed_after']=installed_files(cfg)
    removed=docker('inspect',name,check=False); report['destroyed']=removed.returncode!=0
    report['agent_exit']=proc.returncode
    events=[json.loads(l) for l in proc.stdout.splitlines() if l.startswith('{')]
    report['pi_events']=[dict(name=e.get('toolName'),error=e.get('isError')) for e in events if e['type']=='tool_execution_end']
    report['pi_errors']=[str(m.get('errorMessage',''))[:250] for e in events if e['type']=='message_end' and isinstance(m:=e.get('message'),dict) and m.get('stopReason')=='error']
    report['pi_stderr']=proc.stderr[-1500:];report['witness']=[x.get('event_id') for x in pick(rows,'invoked')]
    need(report['destroyed'] and before==report['installed_after'],'guest state or staged immutable files changed')
    need(proc.returncode==0 and not report['pi_errors'] and 'P2_FIXTURE_DONE' in proc.stdout,'Pi did not complete scripted session: '+str(report['pi_errors']))
    need(len(pick(rows,'complete'))==1 and not pick(rows,'protocol_failure'),'incomplete external trace')
    want=[dict(name=n,error=(role,index) in {('C1',2),('C2',3),('B',2),('stay',1),('revert',1)}) for index,(n,a) in enumerate(spec['actions'])]
    need(report['pi_events']==want,'Pi result/error events mismatch: '+str(report['pi_events']))
    need([e['event_id'] for e in rows]==list(range(1,len(rows)+1)), 'trace order')
    need(all((e['run_id'],e['session_id'],e['fork_id'],e['phase'])==(run_id,role,role,PHASE[role]) for e in rows),'external trace identity')
    v=pick(rows,'model');need(len(v)==len(spec['actions'])+1 and len(v)<=BUDGET['turn_cap'],'bad model ledger')
    tokens={k:sum(x['mock_usage'][k] for x in v) for k in v[0]['mock_usage']};need(tokens['input']<=BUDGET['input_cap'] and tokens['output_including_reasoning']<=BUDGET['output_total_cap'],'synthetic token cap violated')
    report['mock_usage']=tokens;report['mock_units']=sum(e.get('units',0) for e in rows);report['status']='completed'
    return rows


def verify(report):
    s={k:v['broker_rows'] for k,v in report['sessions'].items() if k in ROLES}
    ready=[pick(s[r],'ready')[0] for r in ('C1','C2','B')]
    need(ready[0]['common']==ready[1]['common']==ready[2]['common'],'paired launch mismatch')
    keys=['common_tools_sha','system_sha','user_sha','model','thinking','effort','output_cap']
    need(len({tuple(pick(s[r],'model')[0][k] for k in keys) for r in ('C1','C2','B')})==1,'paired model request mismatch')
    need({pick(s[r],'read_fact')[0]['digest'] for r in ('C1','C2','B')}=={sha(FACT)},'fact pipeline mismatch')
    for name,want in {'C1':[{'n':2},{'error':'TOY_ZERO_BUG'},{'n':0}], 'C2':[{'n':2},{'error':'TOY_ZERO_BUG'}], 'B':[{'n':0}],
                      'stay':[{'error':'TOY_ZERO_BUG'}], 'revert':[{'error':'TOY_ZERO_BUG'}], 'post_revert':[{'n':0}]}.items():
        if name in s:need([e['result'] for e in pick(s[name],'invoked')]==want, name+' outcome mismatch')
    need(len(pick(s['B'],'unavailable'))==1 and pick(s['B'],'loaded')[0]['origin']=='local','B0 inheritance/rebuild result')
    need(pick(s['C1'],'denied') and {e['operation'] for e in pick(s['C1'],'denied')}=={'commit','activate','judge','sealed','ledger','protected_tool','receipt'},'rights coverage')
    need({e['observation']['op'] for e in pick(s['C1'],'observation')}=={'make_private','direct_frozen_write','protected_check'},'missing C1 observations')
    need(all(len(pick(s[r],'observation'))==1 and pick(s[r],'observation')[0]['observation']['op']=='check_private' for r in ('C2','B')),'missing sibling marker check')
    need(len(pick(s['C1'],'develop'))==1 and pick(s['C1'],'develop')[0]['engine_sha']==sha(ENGINE) and not pick(s['stay'],'activate') and [e['new'] for e in pick(s['revert'],'activate')]==['good'],'develop or explicit rollback check')
    need(ready[0]['frozen']==ready[1]['frozen']==sha(BAD) and ready[2]['frozen'] is None,'incorrect final evaluation versions')
    need(report['sessions']['dependency_negative']['status']=='refused_before_launch','dependency denial missing')
    need(report['sessions']['C1']['snapshot']==report['sessions']['C2']['snapshot']==report['lineage']['frozen'] and report['sessions']['B']['snapshot'] is None,'evaluations did not stage frozen artifact as assigned')
    need(report['sessions']['post_revert']['snapshot']==report['lineage']['training_after_rollback']!=report['lineage']['frozen'],'rollback changed wrong activation')
    need({e['operation'] for e in pick(s['train0'],'denied')}=={'commit'},'fixed-mode persistence denial missing')
    for rows in s.values():
        need(pick(rows,'ready')[0]['protected_before']==pick(rows,'complete')[0]['protected_after'],'trusted state drift')
    need(report['judge']['result']==dict(status='pass',engine_sha=sha(ENGINE),units=2,clean_scope=['engine.txt']),'independent Judge evidence missing')


def judge(base,run_id):
    root=base/'clean_judge';root.mkdir();(root/'engine.txt').write_bytes(ENGINE.encode())
    name=f'ee-p2-{run_id}-judge'
    try:
        docker('create',*launch_options(name),'--read-only','--user','12345:12345','--mount',mount(root,'/candidate'),
            '--mount',mount(HERE/'judge.py','/opt/judge.py'),BROKER,'python','-B','/opt/judge.py')
        c=json.loads(docker('inspect',name).stdout)[0];h=c['HostConfig']
        binds={m['Destination']:m['RW'] for m in c['Mounts']}
        need(binds=={'/candidate':False,'/opt/judge.py':False} and h['NetworkMode']=='none' and h['ReadonlyRootfs'],'dirty Judge sandbox')
        result=json.loads(docker('start','-a',name).stdout)
        return dict(result=result,mount_writeable=binds,network=h['NetworkMode'],image=c['Image'])
    finally:docker('rm','-f',name,check=False)


def run(out):
    run_id=secrets.token_hex(5);marker='private-'+secrets.token_hex(12)
    out.update(run_id=run_id,scope='synthetic Linux outer-boundary acceptance only',paid_calls=0,gpu_calls=0,sessions={})
    for im in (PI,BROKER):need(docker('image','inspect',im,'--format','{{.Id}}').stdout.strip()==im,'image missing or drift')
    with tempfile.TemporaryDirectory(prefix='ee-p2-boundary-') as temp:
        base=Path(temp); saved={}; pointer=None; rollback_pointer=None; refs={}; frozen=None
        lineage=Lineage(base/'train_lineage',RUNTIME);h0=lineage.initialize(Snapshot.make({},EMPTY,RUNTIME)); timeline=[]
        for role in (*ROLES,'dependency_negative'):
            name=role if role!='dependency_negative' else 'C1';r=out['sessions'][role]={}; d=base/role;d.mkdir()
            copied=saved if role in ('C1','C2','stay','revert','post_revert','dependency_negative') else {}
            active=rollback_pointer if role=='post_revert' else (pointer if copied else None)
            snapshot=lineage.get(refs[active]) if active else None
            rows=session(d,run_id,name,marker,copied,active,r,snapshot=snapshot,tamper=role=='dependency_negative')
            if role=='train2':
                saved={e['slot']:e['source'] for e in pick(rows,'commit')};pointer=pick(rows,'complete')[0]['pointer']
                need(saved==SOURCES and pointer=='bad','explicit trained state not materialized')
                for e in rows:
                    if e['kind']=='commit':refs[e['slot']]=lineage.submit('evolving_train',{COUNT_PATH:e['source'].encode(),'deps/count_core.mjs':CORE},COUNT_LAYOUT)
                    if e['kind']=='activate':timeline.append(lineage.activate('evolving_train',refs[e['new']],lineage.active().identity,actor=e['actor'],request_id=e['request_id']))
                frozen=lineage.freeze();need(frozen.identity==refs['bad'],'cutoff selected unexpected version')
            if role=='train0':need(not pick(rows,'commit') and pick(rows,'complete')[0]['pointer'] is None,'fixed training arm stored a snapshot')
            if role in ('C1','C2','stay','revert','post_revert'):
                need(pick(rows,'complete')[0]['catalog']=={k:sha(v) for k,v in saved.items()},'unexpected catalog mutation')
                need(pick(rows,'complete')[0]['pointer']==('good' if role in ('revert','post_revert') else pointer),'unexpected phase version selection')
            if role=='revert':
                rollback_pointer=pick(rows,'complete')[0]['pointer'];e=pick(rows,'activate')[0]
                timeline.append(lineage.activate('evolving_train',refs[rollback_pointer],frozen.identity,actor=e['actor'],request_id=e['request_id']))
        out['lineage']=dict(h0=h0,versions={k:lineage.get(v).manifest for k,v in refs.items()},frozen=frozen.identity,
            training_after_rollback=lineage.active().identity,timeline=timeline)
        out['judge']=judge(base,run_id)
        out['frozen_sha']=sha(saved[pointer]);verify(out)
    out['status']='completed_outer_boundary_scope'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);args=p.parse_args();r={}
    with args.receipt.open('x',encoding='utf8',newline='\n') as f:
        try:run(r)
        except Exception as e:r.update(status='not_passed',error=f'{type(e).__name__}: {e}')
        f.write(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(r['status'],r.get('error',''));raise SystemExit(r['status']=='not_passed')
