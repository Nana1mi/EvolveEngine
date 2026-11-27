"""Preflight or run one preregistered E0 pair on spark-c3bc. No API calls."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from math import exp, log
import os
from pathlib import Path
import re
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEV = HERE.parent/'v3_dev_rpc'
sys.path.insert(0,str(ROOT))

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_build import BuildLimits, NativeBuildBackend, compare_cold_builds, IMAGE
from evolveengine import v3_level3_eos as eos
from evolveengine.v3_level3_collect import LONG_CTX_CASE, MEMORY_STAGES, QUALITY_CASES, QualityCollector, TimingCollector, source_bytes, write
from evolveengine.v3_level3_deploy import BF16_WORKBUDDY_64K, BF16_WORKBUDDY_64K_V2, WORKBUDDY_PROFILES, GRAPH_PROFILES, DockerService, ServiceSpec
from evolveengine.v3_level3_evidence import (
    DEV_SCORE_RULE, WORKBUDDY_SCORE_RULE, WORKBUDDY_V2_SCORE_RULE, EvidencePolicy, parse_nvidia_smi_device_memory,
)
from evolveengine.v3_level3_gb10 import Gb10, REGISTERED_GPU_LOCK, check_mapping_read, need
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_level3_worker import NativeWorker
from evolveengine.v3_level3_workload import DISTRIBUTION, load_corpus, select_inputs
from experiments.spark26_gguf.level3.calibrate_nvml_pid import verify_calibration


SHA = re.compile('[a-f0-9]{64}')
SCHEMA = 'level3-gb10-baseline-v4'
LEGACY_SCHEMA = 'level3-gb10-baseline-v3'
# Engineering stop guard only, not a per-candidate residency threshold.
E0_MEMORY_GUARD_BYTES = 128*(1<<30)
D3 = 'dc937a78d2c6de9dd87323b62b9a89d159d9dcef77b2adab19f8484b5922f6da'
R1 = '6b13527eaf7796ca42e4d9c66554be2ca09740cd23c7a152d1c87e3e9decd127'
TIMING = {'observe_timing.py':HERE/'observe_timing.py','timing.py':DEV/'timing.py'}
TEXT_TIMING = {**TIMING,'observe_text_timing.py':HERE/'observe_text_timing.py'}
WORKBUDDY_TIMING = {**TIMING,'observe_workbuddy_timing.py':HERE/'observe_workbuddy_timing.py'}
QUALITY = {'observe_quality.py':HERE/'observe_quality.py',**{n:DEV/n for n in ('numeric.py','legacy_eos.py')}}
JUDGE = (DEV/'audit_worker.py',DEV/'node_client.py',DEV/'plan.json',DEV/'model.sha256',
         HERE.parent/'g1_d2'/'audit.py',HERE.parent/'g1_d2'/'probe.py',HERE.parent/'g1_d3'/'screen.py',
         HERE.parent/'g1_dev'/'audit.py',HERE.parent/'reference_pair_audit.py',HERE.parent/'runtime_identity.py',
         HERE.parent/'runtime_residency_audit.py',HERE.parent/'runtime_residency_probe.py',
         HERE.parent/'synthetic_baseline_audit.py')


def fresh(path):
    path = Path(path)
    root = path.parent.resolve(strict=True)/path.name
    root.mkdir(mode=0o700)
    return root


def implementation_pins():
    """Pin the exact hardware-side driver bytes, not just image/workload IDs."""
    modules = ('artifacts','build','chat_timing','components','component_driver','collect','deploy','diagnostic','evidence','eos','long_ctx','trajectory','trajectory_lane','gb10','lifecycle','run','weight_audit','sampling','sampling_lane','scoring',
               'source','startup','upload','worker','worker_channel','worker_spool','holdout','workload')
    module_paths = [ROOT/f'evolveengine/v3_level3_{mod}.py' for mod in modules]
    module_paths.extend((ROOT/'evolveengine/v3_provider.py',ROOT/'evolveengine/v3_runner.py',
                         ROOT/'evolveengine/v3_level3_web_research.py'))
    paths = [HERE/'bf16_controls.py',HERE/'repeat_e0.py',HERE/'calibrate_trajectory.py',HERE/'observe_trajectory.py',HERE/'node_preflight.py',HERE/'worker_stdio.py',HERE/'profile_control.py',HERE/'calibrate_nvml_pid.py',
             HERE/'calibrate_cuda_meter.py',*JUDGE,
             *module_paths,
             ROOT/'evolveengine/journal.py',ROOT/'evolveengine/v3_harness.py']
    return {path.relative_to(ROOT).as_posix():digest(path.read_bytes()) for path in paths}


def checked_inputs(path, expected, *, allow_legacy=False, split_registration=None):
    need(re.fullmatch(SHA,expected) and path.is_file() and path.stat().st_size<=128*1024,'invalid registration pin or size')
    raw = path.read_bytes()
    need(digest(raw)==expected,'registration hash mismatch')
    doc = json.loads(raw)
    need(raw==canonical(doc)+b'\n','registration must be canonical JSON and LF')
    need(type(doc) is dict and doc.get('schema') in ((SCHEMA,LEGACY_SCHEMA) if allow_legacy else (SCHEMA,)),
         'unregistered configuration version')
    registered = doc['schema']==SCHEMA
    expected_keys = {'schema','boot_id','gpu_uuid','lock','archive','model_dir','anchor_d3',
         'anchor_r1','build_image','build_image_id','runtime_image','runtime_image_id','limits','node_ms','observer_sha256',
         'quality_rule_sha256','measure_workload_sha256','memory_calibration','memory_calibration_sha256',
         'native_backend_sha256'}
    bf16_mode = doc.get('model_format') == 'bf16_safetensors'
    if bf16_mode:
        expected_keys.update({'model_format', 'model_manifest', 'anchor_d3_sha256', 'anchor_r1_sha256'})
        need(registered and type(doc.get('model_manifest')) is dict and doc['model_manifest']
             and all(type(k) is str and type(v) is str and SHA.fullmatch(v) for k, v in doc['model_manifest'].items())
             and SHA.fullmatch(str(doc.get('anchor_d3_sha256', '')))
             and SHA.fullmatch(str(doc.get('anchor_r1_sha256', ''))),
             'invalid BF16 model or anchor manifest registration')
    else:
        expected_keys.add('weight')
    if registered:
        expected_keys.add('implementation_sha256')
    if 'launch_profile' in doc:
        expected_keys.add('launch_profile')
        need(registered and type(doc['launch_profile']) is str and doc['launch_profile'] in GRAPH_PROFILES,
             'unregistered launch profile')
    if 'eos_contract' in doc:
        expected_keys.add('eos_contract')
        need(registered and doc['eos_contract'] == eos.PROTOCOL, 'unknown development EOS contract')
    if 'anchor_long_ctx' in doc:
        expected_keys.update({'anchor_long_ctx', 'anchor_long_ctx_sha256', 'long_ctx_cases', 'long_ctx_cases_sha256'})
        need(registered and bf16_mode
             and SHA.fullmatch(str(doc.get('anchor_long_ctx_sha256', '')))
             and SHA.fullmatch(str(doc.get('long_ctx_cases_sha256', ''))),
             'invalid long-context logprob anchor registration')
    if 'timing_text' in doc:
        expected_keys.add('timing_text')
        binding = doc['timing_text']
        need(registered and type(binding) is dict and set(binding) == {'path','corpus_sha256','seed'} and
             all(type(binding[k]) is str and SHA.fullmatch(binding[k]) for k in ('corpus_sha256','seed')),
             'invalid real-text timing registration')
        corpus = Path(binding['path'])
        need(corpus.is_absolute() and corpus.resolve(strict=True) == corpus and corpus.is_file(),
             'real-text timing corpus must be a resolved local asset')
    need(set(doc)==expected_keys,'unregistered configuration fields')
    if split_registration is None:
        if registered:
            need(doc['implementation_sha256']==implementation_pins(),'hardware-side implementation differs from registration')
    else:
        from experiments.spark26_gguf.level3.live_split_quality import verify_overlay
        need(registered and split_registration['baseline']==expected,'split overlay cannot replace the frozen baseline')
        verify_overlay(doc['implementation_sha256'],implementation_pins(),split_registration)
    need(doc['native_backend_sha256']==digest((ROOT/'evolveengine/v3_level3_build.py').read_bytes()),
         'Native build driver differs from registration')
    path_fields = ('lock','archive','model_dir','anchor_d3','anchor_r1','memory_calibration')
    if not bf16_mode:
        path_fields = (*path_fields, 'weight')
    if 'anchor_long_ctx' in doc:
        path_fields = (*path_fields, 'anchor_long_ctx', 'long_ctx_cases')
    for field in path_fields:
        item = Path(doc[field])
        need(item.is_absolute() and item.resolve(strict=True)==item and not any(c in str(item) for c in ',"\n\r'),
             'registration path not a resolved local asset')
    need(IMAGE.fullmatch(doc['build_image']) and isinstance(doc['limits'],dict) and set(doc['limits'])==set(asdict(BuildLimits()))
         and type(doc['node_ms']) is dict and set(doc['node_ms'])=={'develop_build','develop_quality','develop_measure'}
         and all(type(v) is int and 120000 <= v <= 7200000 for v in doc['node_ms'].values())
         and re.fullmatch(SHA,doc['memory_calibration_sha256']),
         'unsupported image or resource limits')
    return doc


def inventory(host, deadline, build_image=None, runtime_image=None):
    data = host.inventory(deadline)
    need(not data['compute_pids'],'registered GPU is occupied; do not reclaim existing work')
    data['uid'] = os.getuid()
    data['lock'] = str(host.lock)
    data['lock_exists'] = host.lock.exists()
    data['quarantine_active'] = host.quarantine.exists() or host.quarantine.is_symlink()
    data['leftover_level3_jobs'] = host.leftover_jobs(deadline)
    checks = {}
    for label, fn in (
        ('external_memory_used_bytes',lambda:parse_nvidia_smi_device_memory(host.memory(deadline),host.gpu_uuid)),
        ('read_own_map_files',check_mapping_read),
        ('registered_lease_usable',host.check_registered_lease),
        *((label,lambda ref=ref:host.image(ref,deadline)) for label,ref in
          (('native_build_image_id',build_image),('runtime_image_id',runtime_image)) if ref is not None),
    ):
        try:
            checks[label] = {'observed':fn()}
        except (ValueError,OSError,RuntimeError) as exc:
            checks[label] = {'not_observed':type(exc).__name__,'reason':str(exc)[:320]}
    return dict(kind='node_preflight_only',**data,checks=checks)


def quality_files(eos_contract, *, text_observer=None):
    need(eos_contract in (None, eos.PROTOCOL), 'unknown development EOS contract')
    obs = text_observer or (HERE/'observe_text_timing.py')
    return QUALITY if eos_contract is None else {**QUALITY,'observe_text_timing.py':obs}


WORKBUDDY_SCENES = {
    'W1': ((2048, 256),),
    'W2': ((16384, 64),),
    'W3': ((12288, 128),) * 8,
    'W4': ((12288, 128),) * 6 + ((16384, 128),) * 2,
}
WORKBUDDY_V2_SCENES = {
    'W1': ((2048, 256),),
    'W2': ((16384, 64),),
    'W3': ((12288, 128),) * 8,
    'W4': ((12288, 128),) * 6 + ((16384, 128),) * 2,
    'W5': ((12288, 128),) * 20,
    'G64K': ((64000, 64),),
}
WORKBUDDY_V2_SCENE_WARM = {'W3': 2, 'G64K': 0}
WORKBUDDY_V2_SCENE_TIMED = {'W3': 4, 'W4': 4, 'W5': 2, 'G64K': 1}
WORKBUDDY_WARM = 1
WORKBUDDY_TIMED = 2


def collector_pins(timing_text=None, *, eos_contract=None, anchor_d3_sha256=None, anchor_r1_sha256=None,
                   launch_profile=None, legacy_wb_observer_name=False,
                   anchor_long_ctx_sha256=None, long_ctx_cases_sha256=None):
    plan = json.loads((DEV/'plan.json').read_bytes())
    wb_v2 = launch_profile == BF16_WORKBUDDY_64K_V2
    wb = launch_profile in WORKBUDDY_PROFILES
    wb_obs = HERE/'observe_workbuddy_timing.py'
    if timing_text is None:
        timing_files = TIMING
    elif wb and legacy_wb_observer_name:
        timing_files = {**TIMING, 'observe_text_timing.py': wb_obs}
    elif wb:
        timing_files = WORKBUDDY_TIMING
    else:
        timing_files = TEXT_TIMING
    t_hash = {n:digest(source_bytes(p)) for n,p in timing_files.items()}
    q_files = quality_files(eos_contract, text_observer=wb_obs if (wb and legacy_wb_observer_name) else None)
    q_hash = {n:digest(source_bytes(p)) for n,p in q_files.items()}
    q_cases = (*QUALITY_CASES, LONG_CTX_CASE) if anchor_long_ctx_sha256 else QUALITY_CASES
    rule = dict(anchor_manifests={'numeric':anchor_d3_sha256 or D3,'eos':anchor_r1_sha256 or R1},judge_sha256={
        p.relative_to(ROOT).as_posix():digest(source_bytes(p)) for p in JUDGE},observer_sha256=q_hash,
        cases=q_cases,memory_stages=MEMORY_STAGES)
    if eos_contract:
        rule['eos_contract'] = eos_contract
        rule['judge_sha256']['evolveengine/v3_level3_eos.py'] = digest(source_bytes(Path(eos.__file__)))
    if anchor_long_ctx_sha256:
        from evolveengine import v3_level3_long_ctx as long_ctx
        rule['long_context'] = dict(anchor_sha256=anchor_long_ctx_sha256, cases_sha256=long_ctx_cases_sha256)
        rule['judge_sha256']['evolveengine/v3_level3_long_ctx.py'] = digest(source_bytes(Path(long_ctx.__file__)))
    scenes = WORKBUDDY_V2_SCENES if wb_v2 else (WORKBUDDY_SCENES if wb else {k:tuple(map(tuple,v)) for k,v in plan['scenes'].items()})
    warm_waves = WORKBUDDY_WARM if wb else plan['warm_waves']
    timed_waves = WORKBUDDY_TIMED if wb else plan['timed_waves']
    score_rule = WORKBUDDY_V2_SCORE_RULE if wb_v2 else (WORKBUDDY_SCORE_RULE if wb else DEV_SCORE_RULE)
    timing = dict(scenes=scenes,warm_waves=warm_waves,timed_waves=timed_waves,observer_sha256=t_hash,
                  score_rule=score_rule)
    if wb:
        timing['workbuddy_sla'] = True
    if wb_v2:
        timing['workbuddy_v2'] = True
        timing['scene_warm_waves'] = dict(WORKBUDDY_V2_SCENE_WARM)
        timing['scene_timed_waves'] = dict(WORKBUDDY_V2_SCENE_TIMED)
    if timing_text is not None:
        timing['input_plan'] = dict(distribution=DISTRIBUTION, corpus_sha256=timing_text['corpus_sha256'],
                                    seed=timing_text['seed'])
    return rule,timing


def preflight(root, build_image=None, runtime_image=None):
    deadline = monotonic()+60
    host = Gb10.discover(deadline)
    report = inventory(host,deadline,build_image,runtime_image)
    write(root/'preflight.json',report)
    print(canonical(report).decode())


def setup_baseline(root, settings, config_pin):
    host = Gb10(settings['gpu_uuid'],settings['lock'])
    # Use the legacy lease namespace from the experiment account. Do not run
    # under another HOME with an unrelated lock, even if it can access Docker.
    need(host.lock==REGISTERED_GPU_LOCK,'not the registered shared lease')
    deadline = monotonic()+60
    report = inventory(host,deadline,settings['build_image'],settings['runtime_image'])
    write(root/'preflight.json',report)
    checks = report['checks']
    need(report['boot_id']==settings['boot_id'] and report['lock_exists'] and not report['quarantine_active'] and
         not report['leftover_level3_jobs'] and
         checks['native_build_image_id'].get('observed')==settings['build_image_id'] and
         checks['runtime_image_id'].get('observed')==settings['runtime_image_id'] and
         checks['read_own_map_files'].get('observed') is True and
         checks['registered_lease_usable'].get('observed') is True,
         'pinned preflight incomplete; no jobs dispatched')
    calibration = verify_calibration(Path(settings['memory_calibration']),settings['memory_calibration_sha256'],report)
    write(root/'memory-calibration.json',calibration)
    text_spec = settings.get('timing_text')
    eos_contract = settings.get('eos_contract')
    bf16_mode = settings.get('model_format') == 'bf16_safetensors'
    wb_v2 = settings.get('launch_profile') == BF16_WORKBUDDY_64K_V2
    wb = settings.get('launch_profile') in WORKBUDDY_PROFILES
    wb_obs = HERE/'observe_workbuddy_timing.py'
    legacy_wb_name = (wb and not wb_v2 and 'observe_workbuddy_timing.py' not in settings['observer_sha256']
                      and settings['observer_sha256'].get('observe_text_timing.py') == digest(source_bytes(wb_obs)))
    d3_pin = settings.get('anchor_d3_sha256') if bf16_mode else None
    r1_pin = settings.get('anchor_r1_sha256') if bf16_mode else None
    lc_anchor = Path(settings['anchor_long_ctx']) if 'anchor_long_ctx' in settings else None
    lc_sha = settings.get('anchor_long_ctx_sha256')
    lc_cases_path = Path(settings['long_ctx_cases']) if 'long_ctx_cases' in settings else None
    lc_cases_sha = settings.get('long_ctx_cases_sha256')
    rule,timing = collector_pins(text_spec, eos_contract=eos_contract, anchor_d3_sha256=d3_pin,
                                 anchor_r1_sha256=r1_pin, launch_profile=settings.get('launch_profile'),
                                 legacy_wb_observer_name=legacy_wb_name,
                                 anchor_long_ctx_sha256=lc_sha, long_ctx_cases_sha256=lc_cases_sha)
    qhash,thash = (digest(canonical(x)+b'\n') for x in (rule,timing))
    need(settings['observer_sha256']=={**timing['observer_sha256'],**rule['observer_sha256']} and
         (qhash,thash)==(settings['quality_rule_sha256'],settings['measure_workload_sha256']),
         'observer, rule, or workload differs from registration')
    d3,r1 = Path(settings['anchor_d3']),Path(settings['anchor_r1'])
    sys.path.insert(0,str(DEV))
    from audit_worker import D3 as judge_d3, R1 as judge_r1, quality, trusted_reference
    need((judge_d3,judge_r1)==(D3,R1),'trusted Judge anchor constants changed')
    manifests = rule['anchor_manifests']
    for folder, name, group in ((d3,'anchor.jsonl','numeric'),(r1,'e0a.jsonl','eos'),(r1,'e0b.jsonl','eos')):
        trusted_reference(folder,name,manifests[group])
    lc_cases_bytes = None
    if lc_anchor is not None:
        from evolveengine import v3_level3_long_ctx as long_ctx
        long_ctx.verify_anchor(lc_anchor, lc_sha)
        long_ctx.verify_cases(lc_cases_path, lc_cases_sha)
        lc_cases_bytes = lc_cases_path.read_bytes()
    plan = json.loads((DEV/'plan.json').read_bytes())
    if bf16_mode:
        model_files = tuple(sorted(settings['model_manifest'].items()))
        spec = ServiceSpec(settings['runtime_image'],settings['runtime_image_id'],Path(settings['model_dir']),model_files,
                           None,None,host.gpu_uuid,settings.get('launch_profile','diagnostic_eager'),model_format='bf16_safetensors')
    else:
        model_files = tuple(tuple(reversed(line.split('  ',1))) for line in (DEV/'model.sha256').read_text().splitlines())
        spec = ServiceSpec(settings['runtime_image'],settings['runtime_image_id'],Path(settings['model_dir']),model_files,
                           Path(settings['weight']),plan['weight'],host.gpu_uuid,settings.get('launch_profile','diagnostic_eager'))
    weights = spec.recheck_model(deadline=monotonic()+60)
    corpus = None
    if text_spec is not None:
        corpus = load_corpus(text_spec['path'], text_spec['corpus_sha256'], weights['tokenizer.json'])
        if not wb:
            select_inputs(corpus, timing['scenes'], timing['warm_waves']+timing['timed_waves'],
                          seed=text_spec['seed'], pair=0)
    contract = (SourceContract.acquire_spark26_bf16_e0(Path(settings['archive'])) if bf16_mode
                else SourceContract.acquire_spark26_e0(Path(settings['archive']),ROOT))
    engine = contract.from_args({'files':{}})
    q_cases = (*QUALITY_CASES, LONG_CTX_CASE) if lc_cases_bytes is not None else QUALITY_CASES
    p = EvidencePolicy(engine.digest,report['worker_id'],report['boot_id'],host.gpu_uuid,weights,{},qhash,q_cases,
                       thash,digest(b'no profiler lane in E0 preflight'),timing['scenes'],timing['warm_waves'],
                       timing['timed_waves'],MEMORY_STAGES,E0_MEMORY_GUARD_BYTES,('not-invoked',),
                       counter_names=('not-invoked',),origin='hardware_collection',memory_source=calibration['source'],
                       score_rule=timing['score_rule'] if wb else None, workbuddy_sla=wb, workbuddy_v2=wb_v2,
                       scene_warm_waves=timing.get('scene_warm_waves'), scene_timed_waves=timing.get('scene_timed_waves'))
    svc = DockerService(spec,docker=host.docker,diagnostics=root/'service-diagnostics')
    backend = NativeBuildBackend(root/'build-store',image=settings['build_image'],limits=BuildLimits(**settings['limits']))
    grade = eos.grade if eos_contract else quality
    q_obs = quality_files(eos_contract, text_observer=wb_obs if legacy_wb_name else None)
    t_obs = (TIMING if corpus is None else
             ({**TIMING, 'observe_text_timing.py': wb_obs} if legacy_wb_name else
              (WORKBUDDY_TIMING if wb else TEXT_TIMING)))
    collectors = {
        'develop_quality':QualityCollector(host,spec,p,q_obs,rule['observer_sha256'],
                                           score=lambda rows:grade(rows,d3,r1,d3_manifest=d3_pin,r1_manifest=r1_pin,
                                                                   long_ctx_anchor=lc_anchor,long_ctx_sha256=lc_sha),
                                           rule=rule,long_ctx_cases_bytes=lc_cases_bytes),
        'develop_measure':TimingCollector(host,spec,p,t_obs,
                                          timing['observer_sha256'],input_plan=timing.get('input_plan'),documents=corpus),
    }
    @contextmanager
    def pinned(until):
        with host.exclusive(until):
            need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==report['boot_id'],'GB10 boot identity changed')
            yield

    write(root/'registration.json',dict(sha256=config_pin,config=settings))
    return contract,engine,backend,svc,pinned,collectors,timing,calibration


def baseline(root, settings, config_pin):
    contract,engine,backend,svc,pinned,collectors,timing,calibration = setup_baseline(root,settings,config_pin)
    native_builds = []
    metrics = {}
    phases = {}
    results = []
    for arm in ('e0a','e0b'):
        worker = NativeWorker(root/arm,contract,backend,svc,exclusive=pinned,collect=lambda op,*a:collectors[op](op,*a),
                              independent_cold=True)
        build = None
        for op, expected in (('develop_build','built'),('develop_quality','valid'),('develop_measure','observed')):
            result = worker(op,engine,build,settings['node_ms'][op])
            write(root/f'{arm}-{op}.json',asdict(result))
            need(worker.verify(result) and result.status==expected,'baseline lane incomplete; do not generate a score')
            if op=='develop_build':
                build = result
                native_builds.append(worker.records[result.evidence][2])
            elif op=='develop_measure':
                metrics[arm] = result.data['scenes']
            elif op=='develop_quality':
                phases[arm] = result.data['observed_peak_sample_bytes']
            results.append(result.evidence)
    reproducibility = compare_cold_builds(*native_builds)
    dev_ratios = {s:metrics['e0b'][s]['dev_score']/metrics['e0a'][s]['dev_score'] for s in timing['scenes']}
    ratios = {scene:metrics['e0b'][scene]['output_toks_per_s']/metrics['e0a'][scene]['output_toks_per_s']
              for scene in timing['scenes']}
    write(root/'pair.json',dict(kind='engineering_E0_E0_not_candidate_or_holdout',engine=engine.digest,
                               receipts=results,build_artifacts=asdict(reproducibility),scenes=metrics,
                               dev_score_rule=DEV_SCORE_RULE,dev_score_ratio_e0b_over_e0a=dev_ratios,
                               equal_scene_dev_score_geomean_e0b_over_e0a=exp(sum(map(log,dev_ratios.values()))/len(dev_ratios)),
                               observed_throughput_ratio_e0b_over_e0a=ratios,
                               equal_scene_geomean_e0b_over_e0a=exp(sum(map(log,ratios.values()))/len(ratios)),
                               observed_peak_compute_pid_bytes=phases, memory_calibration=calibration,
                               candidate_residency_gate=False,noise_band_established=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    inv = sub.add_parser('inventory',help='read-only GPU/daemon checks, no model service or jobs')
    inv.add_argument('new_report_dir',type=Path)
    inv.add_argument('--build-image')
    inv.add_argument('--runtime-image',default=json.loads((DEV/'plan.json').read_bytes())['image'])
    pins = sub.add_parser('pins',help='show the current collector hashes, without providing approval')
    pins.add_argument('--eos-contract', choices=[eos.PROTOCOL])
    pins.add_argument('--timing-text', nargs=2, metavar=('CORPUS_SHA256','SEED'))
    trial = sub.add_parser('baseline',help='explicitly run one pinned E0/E0 pair; no Agent, API, or Q/T')
    trial.add_argument('new_report_dir',type=Path)
    trial.add_argument('--registration',type=Path,required=True)
    trial.add_argument('--sha256',required=True)
    args = parser.parse_args()
    if args.mode == 'pins':
        text_spec = None
        if args.timing_text:
            need(all(SHA.fullmatch(value) for value in args.timing_text), 'expected hex corpus hash and seed')
            text_spec = dict(zip(('corpus_sha256','seed'), args.timing_text))
        rule,timing = collector_pins(text_spec, eos_contract=args.eos_contract)
        print(canonical(dict(quality_rule=rule,quality_rule_sha256=digest(canonical(rule)+b'\n'),timing=timing,
                             measure_workload_sha256=digest(canonical(timing)+b'\n'),
                             implementation_sha256=implementation_pins())).decode())
        return
    os.umask(0o077)
    if args.mode == 'inventory':
        preflight(fresh(args.new_report_dir),args.build_image,args.runtime_image)
        return
    settings = checked_inputs(args.registration,args.sha256)
    root = fresh(args.new_report_dir)
    try:
        baseline(root,settings,args.sha256)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_optimization_claim',kind=type(exc).__name__,
                                         reason=str(exc)[:400]))
        raise


if __name__=='__main__':
    main()
