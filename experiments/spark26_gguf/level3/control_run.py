"""Trusted Native control runner, separate from Agent and sealed evaluations.

Freeze a plan from a completed, hash-pinned engineering E0/E0 pair before
using the explicit run command. A positive verdict here does not prove that
its altered Native path executed; that requires the separate profile lane.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from experiments.spark26_gguf.level3 import native_controls, node_preflight as node
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_build import BuildReceipt, BuildRequest, FileHash, verify_artifacts
from evolveengine.v3_level3_collect import QUALITY_CASES, write
from evolveengine.v3_level3_deploy import stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_level3_worker import NativeWorker, WorkerResult

SCHEMA = 'spark26-native-controls-v1'
PAIR_KIND = 'engineering_E0_E0_not_candidate_or_holdout'
REPLAYED_QUALITY_KIND = 'engineering_E0_E0_quality_replayed_with_verified_wheels'
ARCHIVE = 'archived_control_builds'
MIB = 1 << 20
MARGIN_BYTES = 8*MIB
PLAN_FILES = (HERE/'control_run.py',HERE/'native_controls.py',HERE/'node_preflight.py',
              *(ROOT/'evolveengine'/f'v3_level3_{role}.py' for role in ('build','collect','deploy','evidence','gb10','source','worker')))

# Prespecified order and costs. A failed or missing stage is never repaired by
# substituting another candidate, build, timing window, or threshold.
ORDER = {
    'native_positive': ('build_cold','quality_cold','measure_cold','build_incremental','quality_incremental','measure_incremental'),
    'native_compute_negative': ('build_cold','quality_cold'),
    'residency_negative': ('build_cold','quality_cold'),
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def _sha(s):
    return type(s) is str and node.SHA.fullmatch(s) is not None


def _int(s):
    return type(s) is int and s > 0


def read_pinned(path, pin, cap=256*1024):
    need(_sha(pin) and path.resolve(strict=True)==path and path.is_file() and path.stat().st_size<=cap,
         'registered input path or size is invalid')
    raw = path.read_bytes()
    need(digest(raw)==pin,'registered input fingerprint changed')
    row = json.loads(raw)
    need(type(row) is dict and canonical(row)+b'\n'==raw,'registration is not canonical JSON')
    return row


def ceiling(peaks):
    need(type(peaks) is dict and set(peaks)=={'e0a','e0b'} and all(_int(x) for x in peaks.values()),
         'missing same-configuration E0 PID memory witnesses')
    high,low = max(peaks.values()),min(peaks.values())
    need(high-low<=MARGIN_BYTES,'E0 sampled memory spread exceeds the prespecified guard; no candidate threshold')
    need(high+MARGIN_BYTES<node.E0_MEMORY_GUARD_BYTES,'invalid externally observed E0 memory scale')
    return high+MARGIN_BYTES


def replayed_peaks(baseline, directory, settings, settings_pin):
    """Read fresh same-profile quality receipts from the reused E0 wheels."""
    receipts = baseline.get('receipts')
    need('timing_text' in settings and baseline.get('registration_sha256') == settings_pin
         and type(receipts) is dict and set(receipts) == {'e0a', 'e0b'}
         and all(_sha(pin) for pin in receipts.values()) and len(set(receipts.values())) == 2,
         'missing replayed quality from both registered E0 arms')
    peaks = {}
    observations = set()
    for arm, pin in receipts.items():
        row = read_pinned(directory/f'quality-{arm}.json', pin, cap=128*1024)
        result = row['result']
        data = result['data']
        need(row.get('arm') == arm and row.get('engine') == baseline['engine']
             and _sha(row.get('deployment')) and _sha(row.get('observation_sha256'))
             and _int(row.get('node_ms')) and row['node_ms'] <= settings['node_ms']['develop_quality']
             and result.get('status') == 'valid' and data.get('case_outcomes') == dict.fromkeys(QUALITY_CASES, 'pass')
             and data.get('memory_source') == baseline['memory_calibration']['source'],
             'replayed E0 quality or memory witness does not match the registration')
        observations.add(row['observation_sha256'])
        peaks[arm] = data.get('observed_peak_sample_bytes')
    need(len(observations) == 2, 'replayed E0 arms share a quality observation')
    return peaks


def archived_controls(ref, settings, variants):
    """Locate already-built Native controls without adopting their old verdicts."""
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'expected pinned Native build index')
    index = read_pinned(Path(ref['path']), ref['sha256'], cap=65536)
    requested = {f'{role}/{stage}' for role, stages in ORDER.items() for stage in stages if stage.startswith('build_')}
    need(index and index.keys() <= requested, 'unknown archived control build')
    build_inputs = ('archive', 'build_image', 'build_image_id', 'limits', 'native_backend_sha256', 'boot_id', 'gpu_uuid', 'lock')
    entries = {}
    for key, binding in index.items():
        role, stage = key.split('/')
        need(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'expected pinned Native control receipt')
        path = Path(binding['path'])
        need(path.name == f'{role}-{stage}.json', 'mislabelled Native control receipt')
        row = read_pinned(path, binding['sha256'])
        prior_config = json.loads((path.parent/'registration.json').read_bytes())
        recorded = json.loads((path.parent/'control-plan.json').read_bytes())
        prior_plan = recorded['plan']
        need(recorded['sha256'] == digest(canonical(prior_plan)+b'\n')
             and prior_plan['baseline_registration_sha256'] == prior_config['sha256'] == digest(canonical(prior_config['config'])+b'\n')
             and canonical(prior_plan['schedule']) == canonical(ORDER)
             and prior_plan['candidates'].keys() == ORDER.keys()
             and prior_plan['candidates'][role] == variants[role].digest
             and all(prior_config['config'][name] == settings[name] for name in build_inputs)
             and row.get('engine') == variants[role].digest and row.get('status') == 'built'
             and _sha(row.get('evidence')), 'archived control build uses a different Native environment or source')
        entries[key] = (path, row)
    positive = 'native_positive/'
    cold, incremental = entries.get(positive+'build_cold'), entries.get(positive+'build_incremental')
    need((cold is None and incremental is None) or
         (cold is not None and incremental is not None and cold[0].parent == incremental[0].parent),
         'archived positive builds must preserve the original cold and incremental pair')
    return entries


def restore_control_build(path, row, role, stage, contract, engine, backend):
    """Verify the raw build, source recipe, and both wheel bytes before reuse."""
    directory = path.parent
    result = WorkerResult(**row)
    data = result.data
    mode = stage.removeprefix('build_')
    need(mode in ('cold', 'incremental') and result.operation == 'develop_build'
         and result.engine == engine.digest and result.status == 'built' and result.build is None
         and _sha(result.evidence) and data['build_mode'] == mode
         and _int(result.node_ms) and _int(data['build_ms']) and data['build_ms'] <= result.node_ms
         and data['environment_sha256'] == backend.environment_sha256, 'archived control build identity differs')
    hits = [p for p in (directory/role).glob('job-*/result.json')
            if p.stat().st_size < 65536 and file_sha256(p) == result.evidence]
    need(len(hits) == 1, 'archived build must bind one raw Worker result')
    raw = read_pinned(hits[0], result.evidence)
    need(raw == dict({k: v for k, v in row.items() if k != 'evidence'}, deployment=data['deployment'],
                     observation_pin=None, observation_unavailable=False), 'archived raw control build differs')
    job = hits[0].parent
    metadata = json.loads((job/'native-build.json').read_bytes())
    request = BuildRequest.from_contract(contract, engine, mode=mode)
    source = backend._source_identity(request, monotonic()+60)
    job_name = metadata['job']
    need(type(job_name) is str and job_name.startswith('ee-native-') and len(job_name) == 42
         and all(c in '0123456789abcdef' for c in job_name[10:])
         and metadata['source'] == source and metadata['image'] == backend.image
         and metadata['executor'] == 'local_docker' and metadata['status'] == 'built'
         and metadata['limits'] == asdict(backend.limits) and metadata['recipe'] == request.build_args_json.decode()
         and (metadata['cache_in'] is None if mode == 'cold' else _sha(metadata['cache_in']))
         and (metadata['cache_out'] is None or _sha(metadata['cache_out'])), 'archived Native build provenance differs')
    artifact_dir = directory/'build-store'/'artifacts'/job_name
    need(artifact_dir.resolve(strict=True) == artifact_dir, 'archived Native artifact path redirected')
    native = BuildReceipt(job_name, engine.digest, source, backend.environment_sha256, backend.image, mode, 'built',
                          data['build_ms'], metadata['cache_in'], metadata['cache_out'],
                          artifacts=tuple(FileHash(**entry) for entry in metadata['artifacts']), artifact_dir=artifact_dir,
                          limits=backend.limits, build_args_json=request.build_args_json, executor_kind='local_docker')
    need(native.artifact_hashes == data['wheels'], 'archived wheel hashes differ from control receipt')
    verify_artifacts(native)
    return job, result, native


def attach_control_build(worker, archive, role, stage, contract, engine, backend):
    path, row = archive
    job, result, native = restore_control_build(path, row, role, stage, contract, engine, backend)
    deployment = stage_deployment(native, worker.root/f'archived-{stage}-deployment')
    need(deployment.identity == result.data['deployment'], 'archived control deployment not reproducible')
    worker.records[result.evidence] = (job, result, native, deployment, None, None, False)
    worker.deployments[result.evidence] = deployment
    return result


def draft(settings, settings_pin, pair_path, pair_pin, variants):
    pair = read_pinned(pair_path,pair_pin)
    registration = json.loads((pair_path.parent/'registration.json').read_bytes())
    need(registration=={'sha256':settings_pin,'config':settings},'E0 pair used a different registration')
    calibration = pair.get('memory_calibration')
    need(pair.get('engine')==variants['e0'].digest and pair.get('candidate_residency_gate') is False
         and type(calibration) is dict and calibration.get('source')=='nvidia_smi_compute_pid'
         and calibration.get('result_sha256')==settings['memory_calibration_sha256'],
         'E0 pair lacks registered engine or device memory evidence')
    if pair.get('kind') == REPLAYED_QUALITY_KIND:
        peaks = replayed_peaks(pair, pair_path.parent, settings, settings_pin)
    else:
        receipts = pair.get('receipts', ())
        need(pair.get('kind')==PAIR_KIND and type(receipts) is list and len(receipts)==6
             and all(_sha(pin) for pin in receipts) and len(set(receipts))==6,
             'baseline is not a complete fixed-engine paired hardware run')
        peaks = pair['observed_peak_compute_pid_bytes']
    archive = pair.get(ARCHIVE)
    entries = {}
    if archive is not None:
        need(pair['kind'] == REPLAYED_QUALITY_KIND, 'archived controls require fresh same-profile E0 quality')
        entries = archived_controls(archive, settings, variants)
    maxima = dict(settings['node_ms'])
    max_total = sum(maxima['develop_'+stage.split('_',1)[0]] for role, stages in ORDER.items() for stage in stages
                    if f'{role}/{stage}' not in entries)
    replay = {ARCHIVE: archive} if archive is not None else {}
    return dict(schema=SCHEMA,baseline_registration_sha256=settings_pin,baseline_pair=str(pair_path),pair_sha256=pair_pin,
                candidates={name:variants[name].digest for name in ORDER},schedule=ORDER,
                witness_symbol=native_controls.KERNEL_WITNESS,
                rule=dict(peak_baseline_bytes=peaks,margin_bytes=MARGIN_BYTES,sample_ceiling_bytes=ceiling(peaks),
                          memory_source=calibration['source'],calibration_sha256=settings['memory_calibration_sha256'],
                          compute_failure_case='numeric_32',quality_cases=QUALITY_CASES,profile_required_separately=True),
                implementation={p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in PLAN_FILES},
                node_ms_caps=maxima,max_total_node_ms=max_total,paid_api_calls=0,holds_no_qt=True,**replay)


def check_quality(role, result, threshold):
    """Only trust completed raw-evidence verdicts; missing is never rejection."""
    need(result.operation=='develop_quality','not a quality receipt')
    data = result.data
    cases,peak = data.get('case_outcomes'),data.get('observed_peak_sample_bytes')
    need(type(cases) is dict and cases.keys()==set(QUALITY_CASES) and set(cases.values())<= {'pass','fail'} and
         _int(peak),'missing verified output or PID memory evidence')
    if role=='native_positive':
        need(result.status=='valid' and set(cases.values())=={'pass'} and peak<=threshold,
             'legal Native control did not pass normal quality and residency')
    elif role=='native_compute_negative':
        need(result.status=='violation' and cases['numeric_32']=='fail' and peak<=threshold,
             'Native compute fault was not isolated as a numeric failure')
    elif role=='residency_negative':
        need(result.status=='violation' and set(cases.values())=={'pass'} and peak>threshold,
             'resident BF16 control did not isolate the external memory violation')
    else:
        raise ValueError('unregistered control')


def execute(root, settings, settings_pin, plan, plan_pin):
    contract,e0,backend,service,exclusive,collectors,_,calibration = node.setup_baseline(root,settings,settings_pin)
    variants = native_controls.make(contract)
    actual = draft(settings,settings_pin,Path(plan['baseline_pair']),plan['pair_sha256'],variants)
    need(canonical(actual)==canonical(plan) and variants['e0']==e0 and calibration['source']==plan['rule']['memory_source'],
         'control registration no longer matches node inputs, source, or code')
    limit = plan['rule']['sample_ceiling_bytes']
    collectors['develop_quality'].policy = replace(collectors['develop_quality'].policy,memory_limit_bytes=limit)
    write(root/'control-plan.json',dict(sha256=plan_pin,plan=plan))
    receipts = {}
    receipt_files = {}
    failed_quality_checks = {}
    node_ms = archived_node_ms = 0
    archive = archived_controls(plan[ARCHIVE], settings, variants) if ARCHIVE in plan else {}
    for role,stages in ORDER.items():
        worker = NativeWorker(root/role,contract,backend,service,exclusive=exclusive,collect=lambda op,*a:collectors[op](op,*a))
        build = None
        cold_cache = None
        for stage in stages:
            op = 'develop_'+stage.split('_',1)[0]
            key = f'{role}/{stage}'
            if key in archive:
                result = attach_control_build(worker, archive[key], role, stage, contract, variants[role], backend)
                archived_node_ms += result.node_ms
            else:
                result = worker(op,variants[role],build,plan['node_ms_caps'][op])
                node_ms += result.node_ms
            receipt_files[f'{role}/{stage}'] = write(root/f'{role}-{stage}.json',asdict(result))
            need(worker.verify(result),'control receipt cannot be independently revalidated')
            receipts[f'{role}/{stage}'] = result.evidence
            if op=='develop_build':
                expected = stage.removeprefix('build_')
                need(result.status=='built' and result.data['build_mode']==expected,'control Native build failed')
                native = worker.records[result.evidence][2]
                if expected=='cold':
                    need(native.cache_input_sha256 is None,'unexpected cache reuse in cold control')
                    cold_cache = native.cache_output_sha256
                else:
                    need(_sha(cold_cache) and native.cache_input_sha256==cold_cache,
                         'incremental build did not use the verified cold cache')
                build = result
            elif op=='develop_quality':
                try:
                    check_quality(role,result,limit)
                except ValueError as exc:
                    failed_quality_checks[key] = str(exc)
                    # Independent fixtures still run, without timing the failed one.
                    break
            else:
                need(role=='native_positive' and result.status=='observed' and set(result.data['scenes'])=={'W1','W2','W3','W4'},
                     'positive control timing is incomplete')
    replay = {ARCHIVE: plan[ARCHIVE], 'archived_build_node_ms': archived_node_ms} if archive else {}
    keep = dict(kind='engineering_controls_not_final_admission',receipts=receipts,receipt_files=receipt_files,node_ms=node_ms,
                sample_ceiling_bytes=limit,native_path_execution_verified=False,noise_band_established=False,
                holds_no_qt=True,paid_api_calls=0,**replay)
    if failed_quality_checks:
        write(root/'failed-controls.json',dict(keep, failed_quality_checks=failed_quality_checks))
        raise ValueError('one or more registered Native quality controls failed')
    return write(root/'result.json',keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode',required=True)
    for kind in ('bind-builds','plan','run'):
        p = sub.add_parser(kind)
        p.add_argument('--registration',required=True,type=Path)
        p.add_argument('--registration-sha256',required=True)
        if kind in ('bind-builds', 'plan'):
            p.add_argument('--pair',required=True,type=Path)
            p.add_argument('--pair-sha256',required=True)
        if kind == 'bind-builds':
            p.add_argument('--build-index', required=True, type=Path)
            p.add_argument('--build-index-sha256', required=True)
            p.add_argument('--output', required=True, type=Path)
        if kind == 'run':
            p.add_argument('report_dir',type=Path)
            p.add_argument('--plan',required=True,type=Path)
            p.add_argument('--plan-sha256',required=True)
    args = ap.parse_args()
    os.umask(0o077)
    settings = node.checked_inputs(args.registration,args.registration_sha256)
    if args.mode in ('bind-builds', 'plan'):
        variants = native_controls.make(SourceContract.acquire_spark26_e0(Path(settings['archive']),ROOT))
    if args.mode == 'bind-builds':
        pair = read_pinned(args.pair, args.pair_sha256)
        need(pair.get('kind') == REPLAYED_QUALITY_KIND and ARCHIVE not in pair
             and args.output.parent == args.pair.parent, 'archive binding requires a new file beside the E0 quality result')
        draft(settings, args.registration_sha256, args.pair, args.pair_sha256, variants)
        ref = dict(path=str(args.build_index), sha256=args.build_index_sha256)
        archived_controls(ref, settings, variants)
        print(write(args.output, dict(pair, **{ARCHIVE: ref})))
        return
    if args.mode=='plan':
        sys.stdout.buffer.write(canonical(draft(settings,args.registration_sha256,args.pair,args.pair_sha256,variants))+b'\n')
        return
    plan = read_pinned(args.plan,args.plan_sha256)
    need(plan.get('schema')==SCHEMA and plan.get('baseline_registration_sha256')==args.registration_sha256,
         'no matching trusted Native control plan')
    root = node.fresh(args.report_dir)
    try:
        print(execute(root,settings,args.registration_sha256,plan,args.plan_sha256))
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_controls',kind=type(exc).__name__,reason=str(exc)[:400]))
        raise


if __name__=='__main__':
    main()
