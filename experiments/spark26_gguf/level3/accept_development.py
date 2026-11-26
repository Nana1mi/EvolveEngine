"""Private GB10 engineering acceptance. Never opens Q/T, calls a model, or selects an optimization.

Run once in a fresh directory after the registered two-arm and six-pair calibration.
Archives every operation before inspecting the expected outcome. No replacement draws.
"""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import json
import os
import sys
from time import monotonic

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.spark26_gguf.level3 import node_preflight as n,control_run,replay_e0,repeat_e0,native_controls,worker_stdio
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector,scripts
from evolveengine.v3_level3_sampling_lane import SamplingLane
from evolveengine import v3_level3_eos as eos,v3_level3_weight_audit as weights,v3_level3_scoring as scoring
from evolveengine.v3_level3_startup import PATH


STEPS=('graph_e0_replay','replay_e0_build','radix_mmvq_build','radix_mmvq_quality','radix_mmvq_measure',
       'independent_sampling_positive','native_fault_build','native_fault_quality','mirror_build','mirror_quality_and_audit')
SCHEMA='development-controls-after-trajectory-v1'


def checked_noise(noise,traj,reg,pin):
    calc=repeat_e0.summarize(noise['receipts']); n.need(noise['registration_sha256']==reg and
        noise['trajectory_calibration_sha256']==pin and traj['registration_sha256']==reg and calc['stable'] and
        all(noise[k]==v for k,v in calc.items()) and noise['eos_contract']==eos.PROTOCOL,
        'current calibration incomplete or unstable; no controls launched')
    return scoring.replay_audit_floors(noise['receipts'])


def run(root,registration,reg,trajectory,tsha,noise_file,nsha):
    cfg=n.checked_inputs(registration,reg);noise=control_run.read_pinned(noise_file,nsha,2<<20)
    traj=control_run.read_pinned(trajectory,tsha);floors=checked_noise(noise,traj,reg,tsha)
    p=control_run.read_pinned(noise_file.parent.parent/'noise-plan.json',noise['plan_sha256']);ref=p['build_registration']
    old=control_run.read_pinned(Path(ref['path']),ref['sha256']); pair=Path(p['baseline']); f=pair.parent/'e0a-develop_build.json'
    plan=write(root/'plan.json',dict(schema=SCHEMA,steps=STEPS,registration=reg,trajectory=tsha,noise=nsha,
            driver=file_sha256(Path(__file__)),cold_pair=p['baseline_sha256'],workload=p['chat_workload']['sha256'],
            sampling='new E0/reference and trial, alpha=.025, repeat only inconclusive, max two looks',
            stop='first incomplete or unexpected outcome, no replacement',holdout=False,paid_model_calls=0))
    graph=n.fresh(root/'graph'); graphpin=replay_e0.run(graph,cfg,reg,f,p['build_quality_measure_pins']['e0a']['develop_build'],
                                                        build_registration=(old,ref['sha256']))
    work=n.fresh(root/'worker'); contract,e0,backend,service,lock,collectors,_,memory=n.setup_baseline(work,cfg,reg)
    controls=dict(replay=str(graph/'replayed.json'),replay_sha256=graphpin,weight_audit=weights.POLICY)
    n.need(worker_stdio.engineering_ceiling(cfg,e0,memory,controls) is None,'relative memory gate still active')
    q=collectors['develop_quality'];q.policy=replace(q.policy,memory_limit_bytes=None); t=collectors['develop_measure']
    n.need(t.enable_natural_eos(p['chat_workload'])==noise['timing_workload_sha256'],'controls must use registered chat workload')
    lane=TrajectoryCollector(q,scripts(n.HERE));lane.activate(dict(path=str(trajectory),sha256=tsha),reg,e0.digest)
    q.enable_trajectory(tsha,lane.code,lambda rows:eos.grade(rows,Path(cfg['anchor_d3']),Path(cfg['anchor_r1']),paired=True))
    worker=worker_stdio.EngineeringWorker(work/'development',contract,backend,service,exclusive=lock,
           collect=lambda op,*a,**kw:collectors[op](op,*a,**kw),settings=cfg,replay_path=graph/'replayed.json',profile_timing=t)
    worker.trajectory_lane=lane;worker.sampling_lane=SamplingLane(worker,q)
    cases=native_controls.make(contract); positive=contract.from_parent(cases['native_positive'],{'files':{PATH:'{"radix-cache":true}'}})
    engines=dict(base=e0.digest,positive=positive.digest,fault=cases['native_compute_negative'].digest,mirror=cases['residency_negative'].digest)
    write(root/'identities.json',engines); receipts={}; charges={}; caps=cfg['node_ms']
    def op(label,name,engine,build,expected,**opts):
        write(root/(label+'-reserved.json'),dict(operation=name,engine=engine.digest,build=build.evidence if build else None,cap=caps[name],**opts));print('started',label,flush=True)
        if 'sample' in opts: result=worker.sampling(engine,build,caps[name],opts['sample'])
        elif 'pair_index' in opts: result=worker.measure(engine,build,caps[name],opts['pair_index'])
        else:result=worker(name,engine,build,caps[name])
        ok=worker.verify(result);receipts[label]=write(root/(label+'.json'),dict(result=asdict(result),verified=ok));charges[label]=result.node_ms
        print('settled',label,result.status,result.node_ms,flush=True);n.need(ok and result.status==expected,'unexpected '+label+' outcome; no retry');return result
    base=op('e0-build','develop_build',e0,None,'built');b=op('positive-build','develop_build',positive,base,'built')
    n.need(b.data['runtime']['argv'].count('--disable-radix-cache')==0 and b.data['runtime']['argv'].count('--disable-decode-cuda-graph')==0,
           'typed startup configuration not active')
    good=op('positive-quality','develop_quality',positive,b,'valid');n.need('sampling_screen' not in good.data and good.data['trajectory_screen']['status']=='valid' and
          good.data.get('weight_observation') and set(good.data['case_outcomes'].values())=={'pass'},'incomplete fast quality screen')
    measured=op('positive-measure','develop_measure',positive,b,'observed',pair_index=0)
    n.need(measured.data['output_authentication']['status']=='verified' and set(measured.data['output_authentication']['agreement'])==floors.keys(),'missing paired output witnesses')
    for i in (1,2):
        for role in ('reference','trial'):
            draw=op('sample-'+str(i)+'-'+role,'develop_quality',positive,b,'sampled',sample=dict(milestone='historical_best',stage=i,role=role))
        if draw.data['decision']!='inconclusive': break
    n.need(draw.data['decision']=='pass','legal sampling control did not pass within registered looks')
    fault=cases['native_compute_negative'];bad=op('fault-build','develop_build',fault,b,'built');badq=op('fault-quality','develop_quality',fault,bad,'violation')
    n.need(badq.data['case_outcomes']['numeric_32']=='fail','numerical fault not caught by numerical gate')
    mirror=cases['residency_negative'];mb=op('mirror-build','develop_build',mirror,base,'built');mq=op('mirror-quality','develop_quality',mirror,mb,'valid')
    scan=weights.static_scan(contract.base,mirror.edits);n.need(mq.data.get('weight_observation') and any('conversion_near_persistent_storage' in x['hints'] for x in scan['locations']),
           'mirror must be observed and flagged for independent review, not rejected by a memory delta')
    scan_pin=write(root/'mirror-scan.json',scan)
    return write(root/'acceptance.json',dict(schema=SCHEMA,plan=plan,registration=reg,trajectory=tsha,noise=nsha,graph=graphpin,
          mirror_scan=scan_pin,identities=engines,receipts=receipts,node_ms=charges,mirror_disposition='known_fixture_flagged_for_review_not_certified',
          steps=STEPS,all_observed=True,paid_model_calls=0,holdout=False))


def verify(root, row, *, reg, trajectory_pin, noise_pin, graph_pin):
    """Controller reads only copied public control receipts; no inference that a mere missing verdict is a rejection."""
    read=control_run.read_pinned
    n.need((row['schema'],row['registration'],row['trajectory'],row['noise'],row['graph'])==(SCHEMA,reg,trajectory_pin,noise_pin,graph_pin)
           and row['steps']==list(STEPS) and row['all_observed'] is True and row['holdout'] is False and row['paid_model_calls']==0,
           'acceptance does not bind the current observation protocol')
    plan=read(root/'plan.json',row['plan']); n.need(plan['driver']==file_sha256(Path(__file__)) and plan['steps']==list(STEPS)
            and (plan['registration'],plan['trajectory'],plan['noise'])==(reg,trajectory_pin,noise_pin),'wrong control driver or protocol')
    names={'e0-build','positive-build','positive-quality','positive-measure','sample-1-reference','sample-1-trial',
           'fault-build','fault-quality','mirror-build','mirror-quality'}
    more={'sample-2-reference','sample-2-trial'}; pins=row['receipts']; n.need(set(pins) in (names,names|more) and
          pins.keys()==row['node_ms'].keys() and len(set(pins.values()))==len(pins),'missing, duplicate or extra controls')
    records={};engines=row['identities'];n.need(set(engines)=={'base','positive','fault','mirror'} and len(set(engines.values()))==4,'control identities overlap')
    for label,pin in pins.items():
        doc=read(root/(label+'.json'),pin,2<<20);r=doc['result']; group=label.split('-')[0]; action=label.split('-')[1]
        wanted='base' if group=='e0' else 'positive' if group=='sample' else group
        expected='sampled' if group=='sample' else 'built' if action=='build' else 'violation' if group=='fault' else 'observed' if action=='measure' else 'valid'
        op='develop_quality' if group=='sample' else 'develop_'+action
        n.need(doc['verified'] is True and (r['engine'],r['operation'],r['status'])==(engines[wanted],op,expected) and
              0<r['node_ms']==row['node_ms'][label] and control_run._sha(r['evidence']),'incomplete or misbound control '+label);records[label]=r
    for k,r in records.items():
        group,action=k.split('-')[:2]
        if action!='build': n.need(r['build']==records[('positive' if group=='sample' else group)+'-build']['evidence'],'wrong candidate binary')
    argv=records['positive-build']['data']['runtime']['argv'];n.need('--disable-radix-cache' not in argv and '--disable-decode-cuda-graph' not in argv and
        '--disable-prefill-cuda-graph' not in argv and argv[argv.index('--stream-interval')+1]=='1' and argv[argv.index('--context-length')+1]=='8192','runtime controls changed')
    pos=records['positive-quality']['data'];n.need(pos['trajectory_screen']['calibration']==trajectory_pin and pos['trajectory_screen']['status']=='valid' and
        set(pos['case_outcomes'].values())=={'pass'} and 'sampling_screen' not in pos and
        records['positive-measure']['data']['output_authentication']['status']=='verified' and records['fault-quality']['data']['case_outcomes']['numeric_32']=='fail',
        'positive/negative controls not specific to their predicates')
    from evolveengine.v3_level3_sampling import decision
    stages=(1,2) if more<=pins.keys() else (1,);seen=set()
    for i in stages:
        a,b=(records[f'sample-{i}-{role}'] for role in ('reference','trial'))
        for r,role in ((a,'reference'),(b,'trial')):
            data=r['data'];n.need(data['challenge']==dict(milestone='historical_best',role=role,stage=i) and
                data['observed_engine']==engines['base' if role=='reference' else 'positive'] and data['service'] not in seen,'duplicate or unbound sampling service');seen.add(data['service'])
        n.need(b['data']['reference']==a['evidence'] and b['data']['decision']==decision(b['data']['test'])==('pass' if i==stages[-1] else 'inconclusive'),'unregistered sampling disposition')
    scan=read(root/'mirror-scan.json',row['mirror_scan']);n.need(records['mirror-quality']['data'].get('weight_observation') and
          any('conversion_near_persistent_storage' in x['hints'] for x in scan['locations']),'expected separate positive telemetry and review flag')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('new_directory',type=Path)
    for k in ('registration','trajectory','noise'):
        ap.add_argument('--'+k,required=True,type=Path);ap.add_argument('--'+k+'-sha256',required=True)
    a=ap.parse_args();os.umask(0o077);root=n.fresh(a.new_directory);start=monotonic()
    try: print(run(root,a.registration,a.registration_sha256,a.trajectory,a.trajectory_sha256,a.noise,a.noise_sha256),flush=True)
    except BaseException as e: write(root/'incomplete.json',dict(error=type(e).__name__,detail=str(e)[:350],elapsed_s=round(monotonic()-start,3)));raise


if __name__=='__main__': main()
