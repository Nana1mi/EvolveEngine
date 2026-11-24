"""Local V3 pre-pilot start check, with execution gated by a separate explicit flag."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2];sys.path.insert(0,str(REPO))
from evolveengine.deepseek import Provider
from evolveengine.v3_harness import Lineage, Snapshot, canonical, digest
from evolveengine import v3_provider
from evolveengine.v3_provider import GENERATION_TRUNCATED, Limits, MaxLease
from evolveengine.v3_runner import Gateway, runtime_for, sandbox
from evolveengine.v3_session import DevReceipt, Session, SourceContract
from node_client import ALLOWED, PLAN, bundle_file, bundle_manifest, connection, invoke, need
from audit_worker import audit, failed_candidate, trusted_reference, R1, D3
from inspect import getsource
from dataclasses import asdict, replace
from time import monotonic
from p3_protocol import exploratory_index, metrics, propose_w, source_identity, validate_handoff
from verify_common import PI,EMPTY

WALL=1200
NODE=800000
UNIT=200000
CAP=Limits(input=200000,output=24000,turns=28,wall_ms=WALL*1000,per_turn_output=8192)
DEV_CALLS=4
TRAIN_MODES={'B0':'fixed_train','B2':'evolving_train'}
PROTOCOL_VERSION='p3-max-output-8192-v1'
LIVE_P3_APPROVED=False
LIVE_HOLD='live P3 on hold after r03; context candidate is offline-only and paid work needs approval'
SMOKE=REPO/'experiments/p2_os_boundary'
TIMING_BASE=HERE.parent/'four_scene_e0/result-20261123.json'
SYSTEM='''Optimize this authorized isolated SGLang/GGUF Q8_0 inference Engine. Your objective is better inference while preserving quality, full vocabulary, weight format, EOS, and request isolation. Work freely within the resource and file limits. You can build local tools/scripts, adjust your approach, and choose whether to update or activate your own persistent Harness for a later session. You are not required to update it or to follow a particular sequence. The external tools measure and sign exact artifacts; do not optimize the test artifacts or alter measurement semantics. Entry examples are capability descriptions, not required steps. A valid candidate may be explicitly delivered without a performance result; do not report a speedup unless you measured it.

The only editable persistent Engine source is python/sglang/srt/layers/quantization/gguf.py. Edit freely there and explore privately. The ee_rpc operations develop_quality, develop_measure and deliver_engine each accept arguments {paths:[relative Engine sources]}, including [] for the current starting bytes. Quality and measurement are separate; a valid slower Engine can be explicitly delivered. Measure returns four distinct metrics: W1 short decode, W2 interactive decode, W3 reciprocal long-context TTFT, and W4 four-lane overall throughput. The exploratory objective is the equal-weight geometric mean of their E0-relative ratios; E0 is freshly measured on the same worker before each session with the identical timing workload. This is a directional signal from one reference boot and one candidate boot, not a noise band or a confirmed speedup. facts returns sanitized previous development evidence, not hidden test samples. No CUDA profiling RPC is registered for this pilot.

You may use /scratch/harness for scripts and other reusable assets. To create a persistent version, harness_submit takes paths relative to /scratch/harness and an ordered load object with keys prompts, skills, extensions, workflows, memory. Dependencies/files may be included under deps without listing them as entries. Use harness_status with empty arguments to read the active version and training version inventory. Committing does not activate; harness_activate requires the exact version and expected currently active identity. Activation is for the next session, does not change current automatic bindings. Change nothing or retain a lawful version if that is best. Uncommitted cache/log/history/private files are discarded. There are four bounded development calls. Stop when there is no better use of the remaining budget.'''
PROMPT='Explore this Engine and decide how to improve it and your future ability to optimize it. Return an honest summary of what you did.'


def record(path):return json.loads(path.read_bytes())


def pilot_profile():
    # P2/r01/r02 evidence uses the immutable 4,096 profile; both P3 arms use this revision.
    return dict(record(SMOKE/'profile.json'),maxTokens=CAP.per_turn_output)


def execution_protocol():
    return dict(version=PROTOCOL_VERSION,profile_sha=digest(canonical(pilot_profile())),limits=asdict(CAP),
                model='deepseek-flash',thinking='enabled',reasoning_effort='max',modes=TRAIN_MODES,
                node_ms=NODE,dev_calls=DEV_CALLS,job_reservation_ms=UNIT,truncation=GENERATION_TRUNCATED,
                launcher_sha=digest(Path(__file__).read_bytes()))


def episode_gateway(arm,lineage,contract,facts,backend,transport,*,remain,node_left,fingerprint=None,parts=()):
    policy=Session(TRAIN_MODES[arm],lineage,contract,facts,backend,wall_ms=remain,node_ms=node_left,
                   dev_calls=DEV_CALLS,reservations={'develop_quality':UNIT,'develop_measure':UNIT})
    model=MaxLease(replace(CAP,wall_ms=remain),transport,fingerprint=fingerprint,expected_prompt_fragments=parts)
    return Gateway(model,policy),model,policy


def provider_diagnostics():
    return dict(version=v3_provider.DIAGNOSTIC_VERSION,provider_source_sha=digest(Path(v3_provider.__file__).read_bytes()))


def readiness(source):
    raw=source.read_bytes();need(digest(raw)==PLAN['gguf_py'],'wrong E0 source');
    files={
      REPO/'experiments/spark26_gguf/g1_v/receipt-v01-20261123.json':'pass_pre_pilot_lab_scope',
      SMOKE/'receipt-03.json':'completed_outer_boundary_scope',
      SMOKE/'shared-freeze-02.json':'pass_shared_runner_fixture',
      HERE/'launcher-mock-05.json':'pass_launcher_mock_scope',
      HERE/'launcher-pair-07.json':'pass_pair_launcher_fixture',
      HERE/'launcher-max-8192-01.json':'pass_offline_max_output',
      HERE/'g1a-03.json':'pass_g1a_pre_pilot_scope',
      TIMING_BASE:'complete',
      SMOKE/'live-max-01.json':'pass_scoped',
    }
    audits={}
    for path,status in files.items():
        data=record(path);need(data['status']==status,'required preflight absent: '+path.name)
        audits[path.relative_to(REPO).as_posix()]=digest(path.read_bytes())
    archive=Path(os.environ['LOCALAPPDATA'])/'EvolveEngine'
    trusted_reference(archive/'spark26-v32-g1-d3.lv2uVT','anchor.jsonl',D3)
    trusted_reference(archive/'g1-dev-fJPjW8','e0a.jsonl',R1)
    trusted_reference(archive/'g1-dev-fJPjW8','e0b.jsonl',R1)
    g1a=record(HERE/'g1a-03.json');node=g1a['common_runner']
    bundle=bundle_manifest();need(bundle_file('bundle.sha256')==bundle and digest(bundle)==g1a['bundle_sha'],'node worker differs from G1-A acceptance')
    need(g1a['missing_phase']=='insufficient_evidence' and
         g1a['cases']['e0']['result']['residency']['status']=='within_observed_scope' and
         g1a['cases']['retained_qkv_bf16']['result']['residency']['status']=='violation', 'G1-A positive/negative/missing evidence incomplete')
    need(node['worker'][0]['result']['residency']['status']=='within_observed_scope','integrated QKV control missed G1-A')
    for accepted in list(g1a['cases'].values())+node['worker']:
        saved=archive/Path(accepted['archive']).name
        need(audit(saved,archive/'spark26-v32-g1-d3.lv2uVT',archive/'g1-dev-fJPjW8')==accepted['result'],
             'archived worker rows no longer reproduce the accepted result')
    need([r['result']['status'] for r in node['worker']]==['valid','observed'] and node['external_policy'][-1]['status']=='selected_by_agent','worker evidence not accepted')
    auth=record(SMOKE/'live-max-01.json')
    need(auth['verified_model_listing']['id']=='deepseek-flash' and auth['verified_model_listing']['name']=='DeepSeek-V4.1-Flash' and [r['prior_tool_contexts'] for r in auth['provider_rows']]==[0,1,2],'real provider evidence incomplete')
    versions={v for r in auth['provider_rows'] for v in r['versions']}
    need(len(versions)==1,'no unique observed provider version');fingerprint=next(iter(versions))
    p=pilot_profile();runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),p,SYSTEM,None)
    pair=record(HERE/'launcher-pair-07.json');mock=record(HERE/'launcher-mock-05.json');cap_test=record(HERE/'launcher-max-8192-01.json')
    protocol=execution_protocol()
    for evidence in (pair,mock,cap_test):
        need(evidence['runtime']==runtime and evidence['execution_protocol']==protocol and
             evidence['provider_diagnostics']==provider_diagnostics(),'pilot runtime or protocol differs from rehearsal')
    need(digest(getsource(run_pair).encode())==pair['orchestration_sha'] and digest((HERE/'p3_protocol.py').read_bytes())==pair['protocol_sha'],'pilot orchestration differs from rehearsal')
    need(cap_test['fixture_sha']==digest((HERE/'rehearse_output_cap.py').read_bytes()),'output-cap fixture changed after rehearsal')
    PI.verify();return dict(status='ready_to_launch_not_executed',source_sha=digest(raw),evidence=audits,provider_fingerprint=fingerprint,
        worker_bundle_sha=digest(bundle),provider_diagnostics=provider_diagnostics(),execution_protocol=protocol,scope='two isolated development-only sessions, one persistent GGUF Python source, no CUDA profile RPC or formal W',
        budget=dict(per_session_wall_s=WALL,per_session_gpu_node_ms=NODE,per_session_dev_calls=DEV_CALLS,controller_e0_boots=1,e0_charge_included=True,
                    input_tokens=CAP.input,output_tokens_including_reasoning=CAP.output,turns=CAP.turns,per_turn_output=CAP.per_turn_output,request_bytes=CAP.request_bytes))


def write_once(path, data):
    with path.open('xb') as f:f.write(canonical(data)+b'\n')


def run_pair(source_bytes,output,model_for,worker,replay,*,fingerprint=None):
    """For each episode, measure E0 before Pi. Never start S2 before verified S1 facts."""
    output.mkdir(exist_ok=False);trusted=output/'trusted';trusted.mkdir()
    profile=pilot_profile();tools=None
    runtime=runtime_for(PI,(SMOKE/'package-lock.json').read_bytes(),profile,SYSTEM,tools)
    lineage=Lineage(trusted/'harness-lineage',runtime);active=lineage.initialize(Snapshot.make({},EMPTY,runtime))
    base={ALLOWED[0]:source_bytes};engine=dict(base);facts={};sessions=[]
    lifecycle=dict(status='stopped',stage='setup',provider_diagnostics=provider_diagnostics(),execution_protocol=execution_protocol(),agent_model=dict(id='deepseek-flash',name='DeepSeek-V4.1-Flash',effort='max',
                   selection_use='not model-selection evidence',fingerprint=fingerprint),w_started=False)
    try:
        for arm in ('S1','S2'):
            lifecycle['stage']=arm+'_e0';began=monotonic();results=[]
            snapshot=lineage.get(active);start_engine=dict(engine)

            def run_job(op,edits,eng,lease,job):
                src=digest(edits.get(ALLOWED[0],source_bytes));job.update(operation=op,engine=eng,source_sha=src)
                r=worker(op,edits,eng,base,lease,job)
                need(r.engine==eng and r.operation==op and type(r.node_ms) is int and 0<r.node_ms<=lease,'invalid worker receipt')
                job.update(receipt_id=r.receipt_id,status=r.status,node_ms=r.node_ms,summary=r.data)
                return r

            baseline=dict(collocated_e0=True);calibration_source=SourceContract(base,ALLOWED,max_files=1,max_bytes=160*1024,max_text_growth=1024,max_new_literal=1024).from_args({'files':{}})
            try:
                cal=run_job('develop_measure',{},calibration_source.digest,UNIT,baseline)
                need(cal.status=='observed','fresh E0 calibration did not settle')
                metrics(cal.data['scenes']);replay(baseline)
            finally:write_once(output/f'{arm}-E0.json',baseline)
            print(f'{arm}: same-worker E0 verified ({cal.node_ms/1000:.3f} node seconds)',flush=True)

            def backend(op,candidate,lease):
                changed=start_engine|dict(candidate.edits)
                edits={k:x for k,x in changed.items() if x!=base[k]}
                job={};results.append(job);r=run_job(op,edits,candidate.digest,lease,job)
                if op=='develop_measure' and r.status=='observed':
                    r=replace(r,data=dict(r.data,performance_dev=exploratory_index(r.data['scenes'],cal.data['scenes'])))
                return r

            remain=round(WALL*1000-(monotonic()-began)*1000);need(remain>=UNIT,'E0 used the episode wall allowance')
            contract=SourceContract(start_engine,ALLOWED,max_files=1,max_bytes=160*1024,max_text_growth=1024,max_new_literal=1024)
            files=dict(snapshot.items);layout=snapshot.manifest['load'];parts=tuple(files[p].decode() for p in layout['prompts']+layout['memory'])
            gateway,model,policy=episode_gateway('B2',lineage,contract,facts,backend,model_for(arm),
                                                remain=remain,node_left=NODE-cal.node_ms,fingerprint=fingerprint,parts=parts)
            lifecycle['stage']=arm+'_agent';print(arm+': free Agent session started',flush=True)
            receipt={};failure=None
            try:receipt['runner']=sandbox(PI,start_engine,snapshot,runtime,profile,SYSTEM,PROMPT,gateway,seconds=remain//1000,tools=tools)
            except (ValueError,RuntimeError,OSError) as e:failure=type(e).__name__
            if model.rows and model.rows[-1]['result']==GENERATION_TRUNCATED:
                failure=GENERATION_TRUNCATED;lifecycle['stop_class']=failure
            elapsed=round((monotonic()-began)*1000)
            oracle_ok=receipt.get('runner',{}).get('status')=='pi_session_complete' and not policy.worker_invalid
            budget_stop=(policy.calls_left==0 or policy.node_limit-policy.spent_node<UNIT or elapsed>=WALL*1000 or len(model.rows)>=CAP.turns or
                         model.spent_input>=CAP.input or model.spent_output>=CAP.output)
            if oracle_ok and policy.delivered:engine.update(dict(policy.delivered.edits))
            active=lineage.freeze().identity
            row=dict(arm=arm,status='completed' if oracle_ok else 'unsettled',failure=failure,start_H=snapshot.identity,next_H=active,
                start_E=source_identity(start_engine),end_E=source_identity(engine),explicitly_delivered=policy.delivered.digest if oracle_ok and policy.delivered else None,
                ingress_facts_sha=digest(canonical(facts)),runtime=runtime,usage=dict(input=model.spent_input,output_including_reasoning=model.spent_output),
                wall_ms=elapsed,budget_stop=budget_stop,node_ms=policy.spent_node+cal.node_ms,calibration=baseline,rows=policy.rows,workers=results,provider=model.rows,provider_rejections=model.rejections,**receipt)
            write_once(output/f'{arm}.json',row);sessions.append(row)
            for n,blob in engine.items():
                dest=trusted/arm/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
            print(arm+': '+row['status'],flush=True)
            if not oracle_ok:raise RuntimeError('pilot did not settle; no automatic retry or continuation')
            if arm=='S1':
                lifecycle['stage']='S1_to_S2_handoff'
                path=output/'S1-to-S2.json'
                try:
                    s1=record(output/'S1.json');saved={n:(trusted/'S1'/n).read_bytes() for n in base}
                    facts=validate_handoff(s1,base,saved,lineage.get(active).identity,replay)
                    need(saved==engine,'in-memory engine differs from validated ingress')
                    write_once(path,dict(status='validated',facts_sha=digest(canonical(facts)),s1_sha=digest((output/'S1.json').read_bytes()),facts=facts))
                except (ValueError,RuntimeError,OSError,TypeError,KeyError) as e:
                    write_once(path,dict(status='blocked_S2_not_started',error=type(e).__name__,reason=str(e)[:220]));raise
                print('S1 handoff validated from durable S1 output; S2 allowed',flush=True)
        lifecycle.update(status='completed',stage='finished')
    except (ValueError,RuntimeError,OSError,TypeError,KeyError) as e:
        lifecycle.update(error=type(e).__name__,reason=str(e)[:220]);raise
    finally:
        write_once(output/'W-budget-proposal.json',propose_w(sessions))
        write_once(output/'P3-status.json',lifecycle)


def start(source,output,confirmed):
    need(LIVE_P3_APPROVED,LIVE_HOLD)
    raw=source.read_bytes();need(digest(raw)==confirmed['source_sha'],'pinned E0 changed after readiness')
    need(confirmed['provider_diagnostics']==provider_diagnostics() and confirmed['execution_protocol']==execution_protocol(),
         'pilot protocol changed after readiness')
    need(not output.exists(),'P3 output path must be new')
    provider=Provider('https://api.deepseek.com','deepseek-flash','DEEPSEEK_API_KEY','DeepSeek-V4.1-Flash',CAP.per_turn_output,fingerprint=confirmed['provider_fingerprint']);provider.preflight()
    archive=Path(os.environ['LOCALAPPDATA'])/'EvolveEngine'
    with connection() as ssh:
        def model_for(_arm):
            return lambda wire,left:provider.request('/chat/completions',payload=wire,timeout=left)

        def load_result(root,rc,op,engine,lease):
            if rc==7:return failed_candidate(root,op,engine,lease)
            need(rc==0,'development node failure')
            return audit(root,archive/'spark26-v32-g1-d3.lv2uVT',archive/'g1-dev-fJPjW8')

        def public(receipt):
            return {k:receipt[k] for k in ('scenes','max_tv','eos_passed','mixed_passed','residency','failure') if k in receipt}

        def worker(op,edits,engine,base,lease,job):
            root,rc=invoke(ssh,op,edits,engine,base,lease);job.update(path=str(root),exit=rc,lease=lease)
            receipt=load_result(root,rc,op,engine,lease)
            return DevReceipt(engine,op,receipt['status'],receipt['node_ms'],receipt['raw_sha'],public(receipt))

        def replay(job):
            root=Path(job['path']).resolve();need(root.parent==archive.resolve() and root.name.startswith('spark26-v3-rpc.'),'invalid trusted DEV archive')
            receipt=load_result(root,job['exit'],job['operation'],job['engine'],job['lease'])
            patch=record(root/'service/patch.json');need(patch['engine']==job['engine'] and patch['operation']==job['operation'] and
                digest((root/'service/candidate-gguf.py').read_bytes())==job['source_sha'],'raw candidate does not bind to live S1 receipt')
            need((receipt['status'],receipt['node_ms'],receipt['raw_sha'],public(receipt))==(job['status'],job['node_ms'],job['receipt_id'],job['summary']),
                 'raw DEV receipt does not reproduce live S1 receipt')
            if job.get('collocated_e0'):need(not patch['files'] and job['source_sha']==PLAN['gguf_py'],'calibration is not immutable E0')

        run_pair(raw,output,model_for,worker,replay,fingerprint=provider.fingerprint)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pinned_gguf_source',type=Path);p.add_argument('--execute-two-free-sessions',type=Path,metavar='NEW_OUTPUT_DIR');a=p.parse_args()
    if a.execute_two_free_sessions is not None and not LIVE_P3_APPROVED:p.error(LIVE_HOLD)
    ready=readiness(a.pinned_gguf_source);print(json.dumps(ready,indent=2),flush=True)
    if a.execute_two_free_sessions is not None:start(a.pinned_gguf_source,a.execute_two_free_sessions,ready)
