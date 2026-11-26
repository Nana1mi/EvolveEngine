"""Non-scoring CUDA trace for a fixed E0 or the trusted positive Native control.

Not a candidate/Agent profiler. Same-UID application code is not an independent
source of trustworthy profiler files, so this lane accepts only fixed fixtures.
"""
import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
import json
import os
import re
from pathlib import Path
from secrets import token_hex
import sqlite3
import subprocess
import sys
from time import monotonic, monotonic_ns, sleep, time_ns
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from experiments.spark26_gguf.level3 import control_run as controls, native_controls, node_preflight as node, replay_e0, worker_stdio
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import DockerObserver, artifact_space, identity, stage_inputs, write, write_rows
from evolveengine.v3_level3_deploy import GRAPH_PROFILES, LAUNCH, WEIGHT, DockerService, StartedService, launch_args, runtime_container_args, stage_deployment, validate_container
from evolveengine.v3_level3_evidence import _columns, _measure, _sqlite_trace, file_sha256
from evolveengine.v3_level3_gb10 import docker_command, docker_env

SCHEMA = 'level3-fixed-fixture-cuda-profile-v1'
NSYS = '/usr/local/bin/nsys'
DURATION = 255
LEASE_MS = 420000
CAPTURE = '/tmp/ee-fixed-trace'
STATUS = CAPTURE+'.exit'
TRACE_OPTIONS = ('profile','--trace=cuda,nvtx','--cpuctxsw=none','--sample=none',f'--duration={DURATION}',
              '--kill=sigterm','--wait=all','--export=sqlite',f'--output={CAPTURE}')
TRACE_ARGS = (*TRACE_OPTIONS, '--', 'python', *LAUNCH)
SOFTWARE_TRACE_OPTIONS = tuple('--trace=cuda,cuda-sw,nvtx' if arg == '--trace=cuda,nvtx' else arg
                               for arg in TRACE_OPTIONS)
KEEP_REPORT = f'"$@"; code=$?; printf "%s\\n" "$code" > {STATUS}; exec sleep 165'
ENTRY = '/bin/sh'
MODES = ('e0', 'native_positive', 'e0_graph')
GRAPH_TRACE = 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'
CLOCK_GUARD_NS = 10_000_000
# docker cp cannot reliably read tmpfs mounts. Only export fixed, bounded files
# from this known fixture's ephemeral namespace using a read-only descriptor.
EXPORT = '''import os, stat, sys
fd=os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
with os.fdopen(fd,'rb') as f:
    st=os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= int(sys.argv[2]):
        raise SystemExit('invalid trace file')
    remaining=st.st_size
    while remaining:
        data=f.read(min(remaining,1<<20))
        if not data:
            raise SystemExit('incomplete trace file')
        sys.stdout.buffer.write(data)
        remaining-=len(data)
    if f.read(1):
        raise SystemExit('trace file changed')
'''


def need(ok, why):
    if not ok:
        raise ValueError(why)


def trace_args(profile, mode):
    if mode == 'native_positive':
        need(profile in GRAPH_PROFILES, 'Native fixture requires a registered runtime profile')
        # The separately profiled control reuses the verified build. Eager CUDA
        # exposes the actual kernel name even when scoring uses CUDA Graph.
        # Stop collection without signaling the server process group. The
        # host removes the disposable container after exporting the trace.
        options = tuple('--kill=none' if arg == '--kill=sigterm' else arg
                        for arg in SOFTWARE_TRACE_OPTIONS)
        return (*options, '--', 'python', *LAUNCH)
    if mode == 'e0':
        need(profile == 'diagnostic_eager', 'E0 kernel control requires the eager runtime profile')
        return TRACE_ARGS
    need(bool(GRAPH_PROFILES.get(profile)),
         'Graph trace requires a registered Graph profile')
    launch = launch_args(SimpleNamespace(launch_profile=profile))
    # Collect device execution for each whole host-launched graph. The GB10
    # hardware-only trace captured host submissions but lacked the device rows.
    return (*SOFTWARE_TRACE_OPTIONS, '--cuda-graph-trace=graph:host-only', '--', 'python', *launch)


def validate_profile_container(raw, deployment, spec, name, cmd):
    conf = raw['Config']
    need(conf['Entrypoint']==[ENTRY] and conf['Cmd']==list(cmd),'profiler launcher drift')
    # Validate all of the ordinary GPU runtime checks against the actual inspect
    # result except the already verified, trusted change of entrypoint and argv.
    declared = dict(raw,Config=dict(conf,Entrypoint=['python'],Cmd=list(launch_args(spec,deployment))))
    return validate_container(declared,deployment,spec,name)


class ProfileService(DockerService):
    """Separate job container; never used by develop_measure or by the Agent."""
    def __init__(self, spec, *, mode, docker, diagnostics):
        super().__init__(spec, docker=docker, diagnostics=diagnostics)
        self.mode=mode
        self.command = ('-c', KEEP_REPORT, 'ee-fixed-profile', NSYS, *trace_args(spec.launch_profile, mode))

    def call_checked(self, deadline, *argv, max_seconds=25):
        timeout = min(max_seconds,deadline-monotonic())
        need(timeout>0,'profiler lease ended')
        result = self._call(argv,timeout)
        need(result.returncode==0 and len(result.stdout)<=1<<20,'Docker profile operation failed')
        return result.stdout

    @contextmanager
    def start(self, deployment, name, deadline):
        spec = self.spec
        command=self.command
        if getattr(deployment,'runtime_config',b''):
            need(self.mode=='e0_graph','engineering runtime declarations require the ordinary Graph profile')
            split=command.index('--',command.index(NSYS))+2  # trusted nsys options, -- python, then server args
            command=(*command[:split],*launch_args(spec,deployment))
        deployment.verify()
        begin = monotonic_ns()
        weights = spec.recheck_model(deadline=deadline)
        checked = monotonic_ns()
        run = lambda *a: self.call_checked(deadline,*a)
        need(run('image','inspect','--format','{{.Id}}',spec.image).decode().strip()==spec.image_id,'runtime image identity drift')
        missing = self._call(('inspect',name),5)
        need(missing.returncode==1 and b'no such' in missing.stderr.lower(), 'profile container name is not fresh')
        args = runtime_container_args(deployment,image=spec.image,name=name,gpu_uuid=spec.gpu_uuid)
        try:
            raw = run('create',*args,'--pull=never','--ipc','private','--no-healthcheck','--restart','no',
                      '--log-driver','local','--log-opt','max-size=8m','--log-opt','max-file=2','--entrypoint',ENTRY,
                      '--mount',f'type=bind,src={spec.model_dir},dst=/model,readonly',
                      '--mount',f'type=bind,src={spec.gguf},dst=/model/{WEIGHT},readonly',spec.image,*command)
            inspected, = json.loads(run('inspect',name))
            cid = validate_profile_container(inspected,deployment,spec,name,command)
            need(type(cid) is str and len(cid)==64 and all(c in '0123456789abcdef' for c in cid)
                 and raw.decode().strip()==cid,'created profile container identity drift')
            run('start',cid)
            yield StartedService(name,cid,deployment,inspected,weights,begin,checked)
        finally:
            try:
                self._retain_diagnostics(name,deadline)
            finally:
                self._remove(name)

    def collect_trace(self, run, output, deadline):
        end = deadline-55
        while monotonic()<end:
            ok = self._call(('exec',run.container_id,'/bin/cat',STATUS),min(10,end-monotonic()))
            if ok.returncode==0:
                (output/'nsys.exit').write_bytes(ok.stdout)
                need(ok.stdout==b'0\n','Nsight profile exited unsuccessfully')
                break
            sleep(.5)
        else:
            raise TimeoutError('Nsight did not complete with a cleanup margin')
        files = []
        for ext, kind in (('nsys-rep','nsys_rep'),('sqlite','nsys_sqlite')):
            spool = output/('copy-'+ext)
            spool.mkdir(mode=0o700)
            limit = artifact_space(spool)
            need(limit > 0, 'no space to retain the profiler artifact and journal')
            command = docker_command(self.docker,'exec','--user','12345:12345',run.container_id,
                                     'python','-I','-c',EXPORT,CAPTURE+'.'+ext,str(limit))
            DockerObserver.capture(command,spool,end,limit,env=docker_env(),artifact=True)
            captured = spool/'observer.stdout'
            need(captured.stat().st_size>0,'missing profiler artifact')
            path = captured.rename(output/('trace.'+ext))
            files.append(dict(kind=kind, name=path.name, bytes=path.stat().st_size, sha256=file_sha256(path)))
        write(output/'development-artifacts.json',dict(schema='level3-fixture-trace-files-v1', files=files))


def kernel_execution(path, expected):
    """A named CUPTI GPU kernel event is required; symbol/file loading is insufficient."""
    count = elapsed = 0
    pattern = re.compile(r'\b'+re.escape(expected)+r'\b')
    table = 'CUPTI_ACTIVITY_KIND_KERNEL'
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        db.execute('PRAGMA trusted_schema=OFF')
        need({'start','end','demangledName'}<=_columns(db,table) and {'id','value'}<=_columns(db,'StringIds'),
             'missing GPU kernel name schema')
        for a,b,name in db.execute(f'SELECT k.start,k.end,s.value FROM {table} k JOIN StringIds s ON s.id=k.demangledName '
                                   'WHERE instr(s.value,?)>0',(expected,)):
            if not pattern.search(name):
                continue
            need(type(a) is int and type(b) is int and 0<=a<b,'invalid named CUDA kernel interval')
            count += 1
            elapsed += b-a
    need(count>0,'the expected Native kernel did not execute in the captured window')
    return dict(symbol=expected,launches=count,device_duration_ns=elapsed)


def clock_offset():
    before = monotonic_ns()
    epoch = time_ns()
    after = monotonic_ns()
    need(after-before < CLOCK_GUARD_NS, 'observation clock sample was interrupted')
    return epoch-after, epoch-before


def graph_execution(path, rows, before, after):
    """Count whole-Graph records and kernel-node records as distinct evidence."""
    offset_min, offset_max = min(before[0], after[0]), max(before[1], after[1])
    need(offset_max-offset_min < CLOCK_GUARD_NS, 'trace and observer clock alignment is unstable')
    windows = [(row['scene'], max(r['first_ns'] for r in row['requests']) + offset_max + CLOCK_GUARD_NS,
                min(r['last_ns'] for r in row['requests']) + offset_min - CLOCK_GUARD_NS)
               for row in rows if row['timed']]
    totals = {scene: dict(graph_executions=0, graph_duration_ns=0,
                          graph_node_executions=0, graph_node_duration_ns=0,
                          graph_api_calls=0, graph_api_successes=0,
                          graph_api_failures=0, graph_api_duration_ns=0)
              for scene, _, _ in windows}
    need(totals, 'missing timed observer windows')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        db.execute('PRAGMA trusted_schema=OFF')
        epoch_table = 'TARGET_INFO_SESSION_START_TIME'
        need({'utcEpochNs'} <= _columns(db, epoch_table), 'missing Nsight session clock schema')
        epochs = db.execute(f'SELECT utcEpochNs FROM {epoch_table}').fetchmany(2)
        need(len(epochs)==1 and type(epochs[0][0]) is int and epochs[0][0]>0, 'ambiguous Nsight session clock')
        epoch = epochs[0][0]
        schemas = []
        graph_columns = _columns(db, GRAPH_TRACE) if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (GRAPH_TRACE,)).fetchone() else set()
        if {'start','end','deviceId','graphId'} <= graph_columns:
            graph_rows = db.execute(f'SELECT start,end,deviceId,graphId FROM {GRAPH_TRACE}')
            schemas.append('nsight_graph_execution_during_timed_decode_v1')
            for start, end, device, graph in graph_rows:
                need(type(start) is int and type(end) is int and 0<=start<=end<=LEASE_MS*1_000_000,
                     'invalid Graph trace interval')
                if start == end or type(device) is not int or not 0 <= device < 2**31-1 or type(graph) is not int or graph <= 0:
                    continue
                for scene, lower, upper in windows:
                    if lower <= epoch+start < epoch+end <= upper:
                        totals[scene]['graph_executions'] += 1
                        totals[scene]['graph_duration_ns'] += end-start
                        break
        runtime_columns = _columns(db, 'CUPTI_ACTIVITY_KIND_RUNTIME')
        string_columns = _columns(db, 'StringIds')
        if {'start','end','returnValue','nameId'} <= runtime_columns and {'id','value'} <= string_columns:
            runtime_rows = db.execute(
                'SELECT r.start,r.end,r.returnValue,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                'JOIN StringIds s ON s.id=r.nameId WHERE s.value GLOB ?',
                ('cudaGraphLaunch*_v[0-9]*',))
            schemas.append('successful_cuda_graph_launch_runtime_api_during_timed_decode_v1')
            for start, end, status, name in runtime_rows:
                need(type(start) is int and type(end) is int and 0 <= start < end <= LEASE_MS*1_000_000
                     and type(status) is int and isinstance(name, str), 'invalid CUDA Graph runtime API interval')
                if not (name.startswith('cudaGraphLaunch_v') or name.startswith('cudaGraphLaunch_ptsz_v')):
                    continue
                for scene, lower, upper in windows:
                    if lower <= epoch+start and epoch+end <= upper:
                        totals[scene]['graph_api_calls'] += 1
                        totals[scene]['graph_api_duration_ns'] += end-start
                        if status == 0:
                            totals[scene]['graph_api_successes'] += 1
                        else:
                            totals[scene]['graph_api_failures'] += 1
                        break
        kernel_columns = _columns(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
        if {'start','end','deviceId','graphId','graphNodeId'} <= kernel_columns:
            kernel_rows = db.execute('SELECT start,end,deviceId,graphId,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL')
            schemas.append('nsight_graph_node_execution_during_timed_decode_v1')
            for start, end, device, graph, node in kernel_rows:
                need(type(start) is int and type(end) is int and 0<=start<=end<=LEASE_MS*1_000_000,
                     'invalid kernel trace interval')
                if start == end or type(device) is not int or not 0 <= device < 2**31-1:
                    continue
                if not ((type(graph) is int and graph > 0) or (type(node) is int and node > 0)):
                    continue
                for scene, lower, upper in windows:
                    if lower <= epoch+start < epoch+end <= upper:
                        totals[scene]['graph_node_executions'] += 1
                        totals[scene]['graph_node_duration_ns'] += end-start
                        break
        need(schemas, 'missing Nsight Graph and kernel activity schema')
        schema = schemas[0] if len(schemas)==1 else 'nsight_graph_and_kernel_activity_during_timed_decode_v1'
    return dict(schema='level3_cuda_graph_observation_v2', evidence_sources=schemas,
                api_launch_semantics='successful cudaGraphLaunch return confirms host submission, not per-node device completion',
                clock_guard_ns=CLOCK_GUARD_NS, observer_utc_offset_bounds_ns=[offset_min, offset_max],
                trace_start_utc_ns=epoch, scenes=totals,
                host_submission_observed=any(row['graph_api_successes'] for row in totals.values()),
                observed=any(row['graph_executions'] or row['graph_node_executions']
                             for row in totals.values()))


def restore_positive(spec, settings, variants, contract, backend):
    result_path = Path(spec['result'])
    summary = controls.read_pinned(result_path, spec['result_sha256'])
    key = 'native_positive/build_cold'
    path = result_path.parent/'native_positive-build_cold.json'
    row = controls.read_pinned(path, summary['receipt_files'][key])
    if controls.ARCHIVE in summary:
        entries = controls.archived_controls(summary[controls.ARCHIVE], settings, variants)
        if key in entries:
            path, archived = entries[key]
            need(archived == row, 'archived Native witness differs from the replayed build')
    _, result, native = controls.restore_control_build(path, row, 'native_positive', 'build_cold',
                                                       contract, variants['native_positive'], backend)
    return native, asdict(result)


def make_plan(settings_pin, inputs, mode, profile='diagnostic_eager'):
    kernel = native_controls.KERNEL_WITNESS if mode=='native_positive' else 'mul_mat_vec_q'
    return dict(schema=SCHEMA,baseline_registration_sha256=settings_pin,mode=mode,inputs=inputs,
                profile_argv=[NSYS,*trace_args(profile, mode)],expected_kernel=None if mode=='e0_graph' else kernel,
                report_storage='stream_to_host_with_disk_reserve',node_ms=LEASE_MS,paid_api_calls=0,scoring=False,holdout=False,
                runner_sha256=file_sha256(Path(__file__).resolve()))


def execute(root, settings, registration_pin, plan):
    mode, inputs = plan['mode'],plan['inputs']
    profile = settings.get('launch_profile', 'diagnostic_eager')
    need(mode in MODES and canonical(plan)==canonical(make_plan(registration_pin,inputs,mode,profile)),
         'profiling plan differs from the fixed fixture contract')
    graph = mode == 'e0_graph'
    contract,e0,backend,_,exclusive,collectors,_,calibration = node.setup_baseline(root,settings,registration_pin)
    if mode in ('e0', 'e0_graph'):
        keys = {'build','build_sha256'}
        if graph:
            keys |= {'build_registration','build_registration_sha256'}
        need(set(inputs)==keys, 'expected archived E0 build input')
        prior_pin = inputs['build_registration_sha256'] if graph else registration_pin
        prior = controls.read_pinned(Path(inputs['build_registration']),prior_pin) if graph else settings
        replay_e0.check_build_compatibility(settings, prior)
        native,r = replay_e0.restore_build(Path(inputs['build']),inputs['build_sha256'],contract=contract,e0=e0,
                     backend=backend,registration=prior,registration_pin=prior_pin)
    else:
        variants = native_controls.make(contract)
        if set(inputs) == {'plan','plan_sha256','result','result_sha256'}:
            worker_stdio.candidate_ceiling(settings,registration_pin,contract,e0,calibration,inputs)
            native,r = restore_positive(inputs,settings,variants,contract,backend)
        else:
            need(set(inputs) == {'plan','plan_sha256','positive_build','positive_build_sha256'},
                 'expected pinned Native controls or fixed positive-only evidence')
            control_plan = controls.read_pinned(Path(inputs['plan']),inputs['plan_sha256'])
            expected = controls.draft(settings,registration_pin,Path(control_plan['baseline_pair']),
                                      control_plan['pair_sha256'],variants)
            need(canonical(control_plan) == canonical(expected)
                 and control_plan['rule']['memory_source'] == calibration['source'],
                 'positive fixture is outside the registered Native control plan')
            path = Path(inputs['positive_build'])
            need(path.name == 'native_positive-build_cold.json', 'positive fixture requires a cold build')
            controls.read_pinned(path.parent/'control-plan.json',digest(canonical(dict(
                sha256=inputs['plan_sha256'],plan=control_plan))+b'\n'))
            row = controls.read_pinned(path,inputs['positive_build_sha256'],cap=65536)
            if controls.ARCHIVE in control_plan:
                source = controls.archived_controls(control_plan[controls.ARCHIVE], settings, variants)
                if 'native_positive/build_cold' in source:
                    path, archived = source['native_positive/build_cold']
                    need(archived == row, 'archived build receipt differs from the positive fixture')
            _, build, native = controls.restore_control_build(path,row,'native_positive','build_cold',
                                                              contract,variants['native_positive'],backend)
            r = asdict(build)
    deploy = stage_deployment(native,root/'deployment')
    need(deploy.identity==r['data']['deployment'],'profile does not use the recorded wheels')
    timing = collectors['develop_measure']
    spec = timing.service_spec
    if mode == 'native_positive':
        spec = replace(spec, launch_profile='diagnostic_eager')
    host = timing.host
    service = ProfileService(spec,mode=mode,docker=host.docker,diagnostics=root/'profile-diagnostics')
    need(list(service.command[3:]) == plan['profile_argv'], 'profiler command differs from the frozen plan')
    job = root/'profile'
    job.mkdir(mode=0o700)
    mounted = root/'client-input'
    files = {**timing.files,'workload.json':timing.workload}
    policy, script = timing.policy, 'observe_timing.py'
    if timing.schedule is not None:
        prompts, hashes = timing.schedule.get(0)
        files['inputs.json'] = canonical(prompts)+b'\n'
        policy = replace(policy,measure_input_hashes=hashes,measure_pair=0)
        script = 'observe_text_timing.py'
    need(not graph or timing.schedule is not None, 'Graph trace requires registered real-text traffic')
    stage_inputs(mounted,files)
    start = monotonic()
    until = start+LEASE_MS/1000
    version = subprocess.run(docker_command(host.docker,'run','--rm','--pull=never','--runtime=runc','--network=none',
                    '--read-only','--user=12345:12345','--pids-limit=32','--memory=256m','--cap-drop=ALL',
                    '--security-opt=no-new-privileges','--entrypoint',NSYS,spec.image_id,'--version'),
                    env=docker_env(),capture_output=True,check=True,timeout=20)
    need(len(version.stdout)<2048 and b'NVIDIA Nsight Systems version' in version.stdout,'missing profiler identity')
    (job/'nsys.version').write_bytes(version.stdout)
    with exclusive(until),service.start(deploy,'ee-l3-job-profile-'+token_hex(7),until) as live:
        before = clock_offset() if graph else None
        observed = timing.observer(live,mounted,job,until-70,script=script)
        after = clock_offset() if graph else None
        if graph:
            write(job/'clock-offset.json',dict(before=before,after=after))
        policy, binding = identity(host,live,spec,job,'develop_profile',policy,until-60,timing.variant)
        need(json.loads((job/'external-identity.json').read_bytes())['loaded_artifacts']==dict(policy.loaded_artifacts),
             'profile loaded artifact differs from the wheel')
        rows = [json.loads(line) for line in observed.splitlines()]
        write_rows(job/'profile-timing.jsonl',binding,rows)
        metadata = dict(binding,workload_sha256=policy.measure_workload_sha256,clock='external_client_monotonic')
        if timing.schedule is not None:
            metadata['pair_index'] = policy.measure_pair
        measures = _measure(metadata,policy,job/'profile-timing.jsonl')
        service.collect_trace(live,job,until)
    trace = job/'trace.sqlite'
    if graph:
        graph_data = graph_execution(trace, rows, before, after)
        write(job/'graph-execution.json',graph_data)
        need(graph_data['observed'], 'no device Graph work observed during timed decode')
        witness = dict(graph=graph_data,launch_profile=profile)
    else:
        kernel = kernel_execution(trace,plan['expected_kernel'])
        timeline, schema_ok, counters, _ = _sqlite_trace(
            trace, LEASE_MS*1_000_000, (), allow_zero_api_duration=mode == 'native_positive')
        witness = dict(kernel=kernel,timeline=timeline,counter_status='not_requested',
                       counter_schema_observed=schema_ok,unexpected_counter_tables=bool(counters),
                       launch_profile=spec.launch_profile, registered_measurement_profile=profile)
    elapsed = int((monotonic()-start)*1000)
    need(elapsed<=LEASE_MS,'profile exceeded node lease')
    return write(root/'result.json',dict(kind='scoped_fixture_profile_not_measure_not_admission',engine=native.candidate_id,
                 deployment=deploy.identity,build_evidence=r['evidence'],node_ms=elapsed,
                 metrics={scene:{k:v for k,v in row.items() if k!='dev_score'} for scene,row in measures['scenes'].items()},
                 archive={p.name:file_sha256(p) for p in job.iterdir() if p.is_file()},
                 hardware_capabilities_credit=False,**witness))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest='action',required=True)
    for action in ('plan','run'):
        a = sp.add_parser(action)
        a.add_argument('--registration',required=True,type=Path)
        a.add_argument('--registration-sha256',required=True)
        if action=='plan':
            a.add_argument('--mode',choices=MODES,required=True)
            a.add_argument('--inputs',required=True,type=Path,help='canonical trusted fixture input JSON')
            a.add_argument('--inputs-sha256',required=True)
        else:
            a.add_argument('report',type=Path)
            a.add_argument('--plan',required=True,type=Path)
            a.add_argument('--plan-sha256',required=True)
    args = ap.parse_args()
    os.umask(0o077)
    cfg = node.checked_inputs(args.registration,args.registration_sha256)
    if args.action=='plan':
        inputs = controls.read_pinned(args.inputs,args.inputs_sha256,cap=65536)
        sys.stdout.buffer.write(canonical(make_plan(args.registration_sha256,inputs,args.mode,cfg.get('launch_profile','diagnostic_eager')))+b'\n')
        return
    plan = controls.read_pinned(args.plan,args.plan_sha256)
    root = node.fresh(args.report)
    write(root/'plan.json',dict(sha256=args.plan_sha256,plan=plan))
    try:
        print(execute(root,cfg,args.registration_sha256,plan),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(kind=type(exc).__name__,reason=str(exc)[:400],scoring=False))
        raise


if __name__=='__main__':
    main()
