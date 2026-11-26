"""Full-source Pi guest for Level 3; leaves the historical narrow runtime unchanged."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import secrets
import stat
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, time_ns
import subprocess

from .v3_harness import canonical, digest
from .v3_runner import HERE, RUNNER_FILES, Relay, docker, launch, need, ro

RPC = HERE / 'v3_level3_rpc.mjs'
SCRATCH = '768m'
MEMORY = '1536m'
GUEST_FILES = {name: HERE / name for name in ('v3_guest.mjs', 'v3_paged_read.mjs')}
GUEST_FILES['v3_rpc.mjs'] = RPC
GUEST_DEST = {name: '/opt/ee/' + name for name in GUEST_FILES}


def _code():
    return digest(canonical({p.name: digest(p.read_bytes()) for p in (*RUNNER_FILES, RPC, Path(__file__))}))


def _config(profile, system, tools, resources=None):
    settings = dict(profile=profile, system=system, tools=tools, scratch=SCRATCH, memory=MEMORY,
                    protocol='spark26-level3-guest-v1')
    if resources is not None:
        settings['resources'] = resources
    return digest(canonical(settings))


def runtime_for(images, package_lock, profile, system, tools=None, *, resources=None):
    return dict(pi_version='1.5.2', pi_image=images.pi.removeprefix('sha256:'), node_version='24.13.0',
                package_lock_sha=digest(package_lock), runner_sha=_code(),
                base_config_sha=_config(profile, system, tools, resources))


def _inventory(root):
    tree = {}
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        key = path.relative_to(root).as_posix()
        need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'unexpected guest staging entry')
        if stat.S_ISDIR(mode):
            tree[key] = None
        else:
            with path.open('rb') as handle:
                h = sha256()
                while block := handle.read(1024 * 1024):
                    h.update(block)
            tree[key] = h.hexdigest()
    return tree


def _verify_images(images, runtime):
    images.verify()
    need(images.pi == 'sha256:' + runtime['pi_image'], 'Pi image drift')
    prefix = ('run', '--rm', '--network', 'none', images.pi)
    version = docker(*prefix, 'node', '/opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js',
                     '--version').stdout.decode().strip()
    node = docker(*prefix, 'node', '--version').stdout.decode().strip().removeprefix('v')
    lock = docker(*prefix, 'cat', '/opt/pi/package-lock.json').stdout
    need(version == runtime['pi_version'] and node == runtime['node_version']
         and digest(lock) == runtime['package_lock_sha'], 'Pi or Node runtime drift')
    return version, node


def _containers_removed(*names):
    listing = docker('container', 'ls', '--all', '--format', '{{.Names}}', check=False)
    return listing.returncode == 0 and set(listing.stdout.splitlines()).isdisjoint(
        name.encode('ascii') for name in names)


def _cleanup(run, side, volume, relay, report, work_volume=None):
    docker('rm', '-f', run, check=False)
    docker('rm', '-f', side, check=False)
    try:
        if relay is not None:
            relay.close()
            report.update(relay_events=relay.count, transport_fault=relay.error)
    finally:
        try:
            report['isolated_state_removed'] = _containers_removed(run, side)
        finally:
            report['socket_removed'] = docker('volume', 'rm', volume, check=False).returncode == 0
            if work_volume is not None:
                report['isolated_state_removed'] &= docker('volume', 'rm', work_volume, check=False).returncode == 0


def _run_pi(run, timeout, report):
    """Timestamp Pi tool boundaries as Docker emits them; discard all raw message contents."""
    proc = subprocess.Popen(['docker', 'start', '-a', run], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    origin, utc_origin = monotonic(), time_ns()
    report.update(pi_tool_executions=[], pi_tool_events=[], pi_errors=0)

    def consume():
        with proc.stdout as stream:
            while line := stream.readline(4 << 20):
                if not line.endswith(b'\n'):
                    while line and not line.endswith(b'\n'):
                        line = stream.readline(4 << 20)
                    continue
                stamp = utc_origin + round((monotonic()-origin)*1_000_000_000)
                if not line.startswith(b'{'):
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if not isinstance(row, dict):
                    continue
                event = row.get('type')
                if event in ('tool_execution_start','tool_execution_end'):
                    name, tool_id = row.get('toolName'), row.get('toolCallId')
                    if not isinstance(name,str) or len(name) > 128:
                        continue
                    report['pi_tool_events'].append(dict(event=event, tool=name, utc_ns=stamp,
                            call_sha256=digest(tool_id.encode()) if isinstance(tool_id,str) else None))
                    if event == 'tool_execution_end':
                        report['pi_tool_executions'].append(dict(name=name, error=row.get('isError')))
                if event == 'message_end' and isinstance(row.get('message'),dict) and row['message'].get('stopReason') == 'error':
                    report['pi_errors'] += 1

    reader = Thread(target=consume, name='level3-pi-events', daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
        return proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join()


def sandbox(images, contract, engine, snapshot, runtime, profile, base_system, prompt, gateway, *,
            seconds, tools=None, draft=None, on_interrupt=None, resources=None, dependencies=None):
    """Materialize a verified complete tree; never mount its artifact store or the journal."""
    need(seconds is None or type(seconds) is int and seconds > 0, 'invalid session wall lease')
    need(runtime['runner_sha'] == _code() and runtime['base_config_sha'] == _config(profile, base_system, tools, resources),
         'Level 3 runtime or configuration drift')
    need(tools is None or 'read' in tools, 'paged reader requires read permission')
    start = monotonic()
    version, node = _verify_images(images, runtime)
    snapshot.recheck(runtime)
    load = snapshot.manifest['load']
    files = dict(snapshot.items)
    parts = tuple(files[name].decode() for name in load['prompts'] + load['memory'])
    need(parts == gateway.provider._expected_prompt_fragments, 'unpaired Harness fragments')
    spec = dict(system='\n\n'.join((base_system, *parts)), prompt=prompt, profile=profile, tools=tools,
                objective=resources is not None,
                rpc=True, paged_read=True, token=gateway.token,
                provider_timeout_ms=((seconds + 15)*1000 if seconds is not None else gateway.provider.model_socket_idle_seconds*1000), extensions=load['extensions'], skills=load['skills'],
                templates=[name for name in load['workflows'] if name.endswith('.md')])
    run = 'ee-l3-' + secrets.token_hex(8)
    side, volume = run + '-relay', run + '-sock'
    work_volume = run + '-work' if resources else None
    relay, proc, timed_out, interruption = None, None, False, None
    report = dict(run=run, pi=version, node=node, image=images.pi, snapshot=snapshot.identity, engine=engine.digest,
                  wip=None if draft is None else draft.digest)
    with TemporaryDirectory(prefix='ee-level3-input-') as folder:
        root = Path(folder)
        # A draft is optionally checked out into the guest working tree. It is never used as the legal Engine pointer.
        contract.materialize(draft if draft is not None else engine, root / 'engine')
        snapshot.stage(root / 'harness', runtime)
        if dependencies is not None:
            from .v3_level3_dependencies import stage
            stage(snapshot, root/'harness', dependencies)
        (root / 'spec.json').write_bytes(canonical(spec))
        before = _inventory(root)
        docker('volume', 'create', volume)
        try:
            relay = Relay(side, images.bridge, volume, gateway)
            relay.wait()
            opts = launch(run)
            opts[opts.index('--memory') + 1] = MEMORY
            mounts = [value for name, source in GUEST_FILES.items() for value in ('--mount', ro(source, GUEST_DEST[name]))]
            temporary = ['/scratch:rw,nosuid,nodev,mode=1777,size=' + SCRATCH, '/tmp:rw,nosuid,nodev,mode=1777,size=64m']
            if resources:
                report['resources'] = resources
                for option, value in (('--memory', f"{resources['memory_mib']}m"), ('--cpus', str(resources['cpus'])),
                                      ('--pids-limit', str(resources['pids']))):
                    opts[opts.index(option) + 1] = value
                temporary = [f"/tmp:rw,nosuid,nodev,mode=1777,size={resources['tmp_mib']}m"]
                docker('volume', 'create', work_volume)
                work_mount = f'type=volume,src={work_volume},dst=/scratch,volume-nocopy'
                docker('run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--cap-add', 'CHOWN',
                       '--security-opt', 'no-new-privileges', '--user', '0:0', '--mount', work_mount,
                       '--entrypoint', 'chown', images.pi, '12345:12345', '/scratch')
                mounts += ['--mount', work_mount]
            docker('create', *opts, '--read-only', '--user', '12345:12345',
                   *(x for entry in temporary for x in ('--tmpfs', entry)),
                   '--mount', ro(volume, '/gate', 'volume'), '--mount', ro(root, '/input'),
                   *mounts, images.pi, 'node', '/opt/ee/v3_guest.mjs', timeout=30)
            view = json.loads(docker('inspect', run).stdout)[0]
            host = view['HostConfig']
            expected = dict.fromkeys(('/input', '/gate', *GUEST_DEST.values()), False)
            if work_volume:
                expected['/scratch'] = True
            need(view['Image'] == images.pi and view['Config']['User'] == '12345:12345'
                 and host['NetworkMode'] == 'none' and host['ReadonlyRootfs'] and not host['Privileged']
                 and not host.get('DeviceRequests') and host['CapDrop'] == ['ALL']
                 and host['SecurityOpt'] == ['no-new-privileges']
                 and set(host['Tmpfs']) == ({'/tmp'} if work_volume else {'/scratch', '/tmp'})
                 and {m['Destination']: m['RW'] for m in view['Mounts']} == expected, 'Level 3 sandbox isolation drift')
            report['isolation'] = dict(network='none', mounts=expected, gpu=False, readonly_root=True)
            left = None if seconds is None else seconds - (monotonic() - start)
            need(left is None or left > 0, 'wall budget exhausted during guest staging')
            try:
                proc = _run_pi(run, left, report)
            except subprocess.TimeoutExpired:
                timed_out = True
        except KeyboardInterrupt as exc:
            interruption = exc
            if on_interrupt is not None:
                try:
                    on_interrupt()
                except Exception as transport_error:
                    raise interruption from transport_error
            raise
        finally:
            try:
                _cleanup(run, side, volume, relay, report, work_volume)
            except Exception as cleanup_error:
                if interruption is not None:
                    raise interruption from cleanup_error
                raise
        report['unchanged_install'] = _inventory(root) == before
    report.setdefault('pi_tool_executions', [])
    report.setdefault('pi_errors', 0)
    report.update(pi_exit=proc.returncode if proc else None,
                  timed_out=timed_out, wall_ms=round((monotonic() - start)*1000),
                  provider_rows=gateway.provider.rows, gateway_events=gateway.events)
    return report
