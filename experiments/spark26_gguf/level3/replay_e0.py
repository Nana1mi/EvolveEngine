"""Recheck runtime/observer changes on a pinned independent E0 wheel build.

Engineering replay only: no trusted score is inferred from an earlier failed
quality record; source and deployed wheel bytes are reverified before use.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from secrets import token_hex
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from experiments.spark26_gguf.level3 import control_run, node_preflight as node
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_build import BuildReceipt, BuildRequest, FileHash, verify_artifacts
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import BF16_WORKBUDDY_64K, IMMEDIATE_GRAPH_WIRE, launch_args, stage_deployment
from evolveengine.v3_level3_evidence import file_sha256, validate_evidence


def restore_build(build_file, build_pin, *, contract, e0, backend, registration, registration_pin, arm='e0a'):
    """Keep prior build cost, bind exact local source, and verify both wheel lanes."""
    row = control_run.read_pinned(build_file,build_pin)
    prior_root = build_file.parent
    node.need(arm in ('e0a','e0b') and build_file.name==f'{arm}-develop_build.json' and
              json.loads((prior_root/'registration.json').read_bytes())=={'sha256':registration_pin,'config':registration},
              'archived build used a different baseline registration')
    public = row['data']
    node.need(row['operation']=='develop_build' and row['engine']==e0.digest and row['status']=='built' and
              row['build'] is None and public['build_mode']=='independent_cold' and
              public['environment_sha256']==backend.environment_sha256 and control_run._sha(row['evidence']),
              'prior receipt is not the pinned independently built E0')
    db_rows = [p for p in (prior_root/arm).glob('job-*/result.json')
               if p.is_file() and p.stat().st_size<=65536 and file_sha256(p)==row['evidence']]
    node.need(len(db_rows)==1,'unable to bind archived build to a unique raw result')
    job = db_rows[0].parent
    raw_row = json.loads(db_rows[0].read_bytes())
    expected_raw = {k:v for k,v in row.items() if k!='evidence'}
    node.need(raw_row==dict(expected_raw,deployment=public['deployment'],observation_pin=None,observation_unavailable=False),
              'archived build raw fields drifted')
    metadata = json.loads((job/'native-build.json').read_bytes())
    job_name = metadata['job']
    request = BuildRequest.from_contract(contract,e0,mode='independent_cold')
    expected_source = backend._source_identity(request,monotonic()+60)
    node.need(isinstance(job_name,str) and len(job_name)==42 and job_name.startswith('ee-native-') and
              all(c in '0123456789abcdef' for c in job_name[10:]) and
              metadata['image']==backend.image and metadata['source']==expected_source and metadata['status']=='built' and
              metadata['limits']==asdict(backend.limits) and metadata['recipe']==request.build_args_json.decode() and
              metadata['executor']=='local_docker' and metadata['cache_in'] is None and metadata['cache_out'] is None,
              'archived Native source, build environment, or isolation is not pinned')
    artifact_dir = prior_root/'build-store/artifacts'/job_name
    node.need(artifact_dir.resolve(strict=True)==artifact_dir,'archived wheel directory redirected')
    receipt = BuildReceipt(job_name,e0.digest,expected_source,backend.environment_sha256,backend.image,
                           'independent_cold','built',public['build_ms'],None,None,
                           artifacts=tuple(FileHash(**x) for x in metadata['artifacts']),artifact_dir=artifact_dir,
                           limits=backend.limits,build_args_json=request.build_args_json,executor_kind='local_docker')
    node.need(receipt.artifact_hashes==public['wheels'],'archived wheel hashes do not match build result')
    verify_artifacts(receipt)
    return receipt,row


def check_build_compatibility(registration, build_settings):
    unchanged = ('archive','build_image','build_image_id','limits','native_backend_sha256','boot_id','gpu_uuid','lock')
    node.need(all(registration[key] == build_settings[key] for key in unchanged),
              'replay may change observation/launch profile but requires the same registered build inputs')


def graph_delta_numeric(raw):
    # The historical bootstrap client kept only the last frame. Keep its probe
    # inputs, cancellation/identity checks and host verdict unchanged, but
    # reconstruct every token when the pinned service sends incremental deltas.
    anchor = b" final=msg['output_ids'];term=m.get('finish_reason')"
    import_anchor = b'VOCAB=131072\n'
    node.need(raw.count(anchor)==raw.count(import_anchor)==1, 'legacy graph probe source drift')
    return raw.replace(import_anchor, b'from observe_text_timing import merge_stream_output\n\n'+import_anchor).replace(
        anchor, b" term=m.get('finish_reason');final=merge_stream_output(final,msg['output_ids'],n,final=term is not None)")


def adapt_graph_delta_quality(quality):
    key = 'numeric.py'
    source = quality.files[key]
    before = quality.rule['observer_sha256']
    node.need('trajectory' not in quality.rule and digest(source)==before[key] and
              digest(quality.files['observe_text_timing.py'])==before['observe_text_timing.py'], 'unexpected graph probe sources')
    patched = graph_delta_numeric(source)
    quality.files = quality.files | {key:patched}
    quality.rule = dict(quality.rule, observer_sha256=before | {key:digest(patched)},
                        graph_delta_adapter=dict(source=before[key], adapter='strict_merge_stream_output_v1'))
    quality.policy = replace(quality.policy, quality_rule_sha256=digest(canonical(quality.rule)+b'\n'))


def run(root, registration, registration_pin, build_file, build_pin, *, build_registration=None, chat=None):
    build_settings, build_registration_pin = build_registration or (registration, registration_pin)
    check_build_compatibility(registration, build_settings)
    write(root/'replay-plan.json',dict(kind='pinned_wheel_replay_not_E0_E0_pair',registration_sha256=registration_pin,
          prior_build_sha256=build_pin,prior_build=str(build_file), prior_build_registration_sha256=build_registration_pin,
          runner_sha256=file_sha256(Path(__file__).resolve()),
          runtime_driver_sha256=file_sha256(ROOT/'evolveengine/v3_level3_deploy.py'),
          launch_profile=registration.get('launch_profile','diagnostic_eager'),
          actions=('develop_quality','develop_measure'),paid_api_calls=0))
    contract,e0,backend,service,exclusive,collectors,*_ = node.setup_baseline(root,registration,registration_pin)
    write(root/'launch-argv.json',dict(profile=service.spec.launch_profile,argv=launch_args(service.spec)))
    if service.spec.launch_profile in (IMMEDIATE_GRAPH_WIRE, BF16_WORKBUDDY_64K):
        adapt_graph_delta_quality(collectors['develop_quality'])
    if chat is not None:
        collectors['develop_measure'].enable_natural_eos(chat)
    native,build = restore_build(build_file,build_pin,contract=contract,e0=e0,backend=backend,
                                 registration=build_settings,registration_pin=build_registration_pin)
    deployment = stage_deployment(native,root/'deployment')
    node.need(deployment.identity==build['data']['deployment'],'wheel deployment is not identical to the archived build')
    write(root/'build-link.json',dict(prior_build=build,sha256=build_pin,reused=True))
    receipts, observations = {}, {}
    for op,expect in (('develop_quality','valid'),('develop_measure','observed')):
        started = monotonic()
        cap = registration['node_ms'][op]
        deadline = started+cap/1000
        archive = root/op
        with exclusive(deadline),service.start(deployment,'ee-l3-job-replay-'+token_hex(8),deadline) as started_service:
            witness = collectors[op](op,started_service,archive,deadline)
        verdict = validate_evidence(archive,witness.policy,{op:witness.pin}).data['operations'][op]
        node_ms = int((monotonic()-started)*1000)
        node.need(node_ms<=cap,'E0 replay exceeded the registered observation time')
        receipts[op] = write(root/(op+'.json'),dict(observation_sha256=witness.pin,build_evidence=build['evidence'],
                             node_ms=node_ms,result=asdict(verdict)))
        observations[op] = asdict(verdict)
        print(op,verdict.status,flush=True)
        node.need(verdict.status==expect,'runtime observation incomplete; not a quality pass or score')
    logs = sorted((root/'service-diagnostics').glob('*.log'))
    stages = ('decode','prefill')
    captures = {phase: [p.name for p in logs if f'Capture target {phase} CUDA graph end.'.encode() in p.read_bytes()]
                for phase in stages}
    return write(root/'replayed.json',dict(kind='trusted_engineering_replay_not_E0_E0_pair',receipts=receipts,
                 engine=e0.digest, profile=service.spec.launch_profile,
                 capture_log_witnesses=captures, logs={p.name:file_sha256(p) for p in logs}, verdicts=observations))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report_dir',type=Path)
    ap.add_argument('--registration',type=Path,required=True)
    ap.add_argument('--registration-sha256',required=True)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--build-sha256',required=True)
    ap.add_argument('--build-registration', type=Path)
    ap.add_argument('--build-registration-sha256')
    ap.add_argument('--chat-workload', type=Path)
    ap.add_argument('--chat-workload-sha256')
    ap.add_argument('--split-registration', type=Path)
    ap.add_argument('--split-registration-sha256')
    args = ap.parse_args()
    if bool(args.build_registration) != bool(args.build_registration_sha256):
        ap.error('prior registration and fingerprint must be supplied together')
    if bool(args.chat_workload) != bool(args.chat_workload_sha256):
        ap.error('chat workload and fingerprint must be supplied together')
    if bool(args.split_registration) != bool(args.split_registration_sha256):
        ap.error('split registration and fingerprint must be supplied together')
    os.umask(0o077)
    split_reg = None
    if args.split_registration:
        split_reg = control_run.read_pinned(args.split_registration, args.split_registration_sha256)
    settings = node.checked_inputs(args.registration,args.registration_sha256,allow_legacy=True,split_registration=split_reg)
    prior = None
    if args.build_registration:
        prior = (control_run.read_pinned(args.build_registration, args.build_registration_sha256),
                 args.build_registration_sha256)
    chat = None
    if args.chat_workload:
        chat = dict(path=str(args.chat_workload.resolve()), sha256=args.chat_workload_sha256)
    root = node.fresh(args.report_dir)
    try:
        print(run(root,settings,args.registration_sha256,args.build,args.build_sha256,build_registration=prior,chat=chat))
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_optimization_claim',kind=type(exc).__name__,reason=str(exc)[:400]))
        raise


if __name__=='__main__':
    main()
