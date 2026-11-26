"""Stage verified Native wheels for a read-only, offline candidate deployment.

The host never imports the wheels, runs pip against them, executes console
scripts, or writes into the runtime image. This stage proves bytes, not that a
GPU process loaded or executed them. The host observer must prove that later.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import stat
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic, monotonic_ns

from .v3_level3_gb10 import docker_command, docker_env
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from .v3_level3_build import BuildError, CleanupUncertain, GIB, WHEEL_NAME, _claim, verify_artifacts
from .v3_harness import canonical, digest

TARGET = '/ee-candidate'
JIT_CACHE = '/ee-jit'
SCRATCH_MIB = 16384
JIT_MIB = 4096
RUNTIME_ENV = {
    'PYTHONPATH':f'{TARGET}:/opt/spark26-deps','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HOME':'/tmp','TMPDIR':'/tmp',
    'XDG_CACHE_HOME':f'{JIT_CACHE}/xdg-cache','CUDA_CACHE_PATH':f'{JIT_CACHE}/cuda-cache',
    'TRITON_CACHE_DIR':f'{JIT_CACHE}/triton-cache','SGLANG_JIT_CACHE_DIR':f'{JIT_CACHE}/sglang-cache',
}


def runtime_tmpfs(scratch_mib=SCRATCH_MIB):
    # Triton loads a generated shared library from its cache. Only its
    # bounded ephemeral cache needs executable mappings; /tmp stays noexec.
    return {'/tmp':f'rw,noexec,nosuid,nodev,mode=1777,size={scratch_mib}m',
            JIT_CACHE:f'rw,exec,nosuid,nodev,mode=1777,size={JIT_MIB}m'}

SCHEMA = 'level3-native-deployment-v1'
MAX_EXPANDED = 6 * GIB
MAX_FILES = 100_000
BLOCK = 1024 * 1024
MAPS = ('sgl_kernel/sm90/common_ops.abi3.so', 'sgl_kernel/sm100/common_ops.abi3.so')


def _need(condition, error):
    if not condition:
        raise BuildError(error)


def _export_root(name, *, lane, dist_info):
    parts = name.split('/')
    if len(parts) >= 3 and parts[0] == dist_info.removesuffix('.dist-info') + '.data':
        if parts[1] not in ('purelib', 'platlib'):
            # Wheels may carry optional executable CLI wrappers or headers.
            # None are part of the candidate runtime mount.
            _need(parts[1] in ('scripts', 'headers', 'data'), 'unknown wheel data kind')
            return None
        parts = parts[2:]
    expected = 'sglang' if lane == 'engine' else 'sgl_kernel'
    _need(len(parts) >= 2 and parts[0] in (expected,dist_info), 'wheel installs outside the approved package')
    _need(not any(x.casefold().endswith(('.pth','.pyc','.pyo')) or x.casefold() == '__pycache__' for x in parts),
          'wheel contains a startup hook or precompiled Python cache')
    return '/'.join(parts)


@dataclass(frozen=True)
class RuntimeFile:
    path: str
    size: int
    sha256: str


def _reparse(st):
    return bool(getattr(st,'st_file_attributes',0) & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0))


def _hash_file(path, expected_size, *, deadline=None):
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and not _reparse(info) and info.st_size == expected_size,
          'invalid deployment file identity')
    h = sha256()
    with path.open('rb') as inp:
        while block := inp.read(BLOCK):
            _need(deadline is None or monotonic() < deadline, 'asset hash deadline expired')
            h.update(block)
    return h.hexdigest()


def _publish_bind_tree(site):
    # The Worker uses umask 077. The bind subtree (not its trusted parent)
    # must be traversable by Docker's unrelated service UID, without write access.
    for directory, _, _ in os.walk(site):
        Path(directory).chmod(0o755)


@dataclass(frozen=True)
class Deployment:
    candidate: str
    build_job: str
    wheels: tuple[tuple[str, str], ...]
    files: tuple[RuntimeFile, ...]
    root: Path
    parent_identity: str | None = None
    # Source-bound by the host Worker. Not a file in the wheel or a container override.
    runtime_config: bytes = b''

    @property
    def tree(self):
        return self.root / 'site'

    def record(self):
        record = dict(schema=SCHEMA, candidate=self.candidate, build_job=self.build_job, wheels=dict(self.wheels),
                      files=[asdict(file) for file in self.files])
        if self.parent_identity is not None:
            record['parent_deployment'] = self.parent_identity
        return record

    @property
    def identity(self):
        return digest(canonical(self.record()))

    def native_mapping(self, variant):
        _need(variant in ('sm90','sm100'), 'unsupported Native mapping variant')
        path = f'sgl_kernel/{variant}/common_ops.abi3.so'
        entry = next((item for item in self.files if item.path == path), None)
        _need(entry is not None, 'runtime lacks pinned Native mapping')
        return {path: f'{TARGET}/{path}'}, {path: entry.sha256}

    def verify_runtime_access(self):
        """Reject an unreadable mount before Python can fall back to image packages."""
        if os.name != 'posix':
            return self
        dirs = {self.tree}
        for entry in self.files:
            path = self.tree/entry.path
            _need(stat.S_IMODE(path.lstat().st_mode) == 0o444,
                  'runtime bind file is not read-only and readable by the service UID')
            dirs.update(self.tree/folder for folder in Path(entry.path).parents)
        for path in dirs:
            _need(stat.S_IMODE(path.lstat().st_mode) == 0o755,
                  'runtime bind directory is not traversable by the service UID')
        return self

    def verify(self):
        _need(self.root.is_dir() and self.root.resolve() == self.root and self.tree.is_dir()
              and not _reparse(self.tree.lstat()) and not self.tree.is_symlink(), 'unsafe deployment root')
        manifest = self.root/'manifest.json'
        data = canonical(self.record())+b'\n'
        _need(_hash_file(manifest,len(data)) == digest(data),'deployment manifest drift')
        remaining = {entry.path:entry for entry in self.files}
        expected_dirs = {parent.as_posix() for f in self.files for parent in Path(f.path).parents if parent != Path('.')}
        for base, dirs, files in os.walk(self.tree,followlinks=False):
            for child in dirs:
                path = Path(base)/child
                rel = path.relative_to(self.tree).as_posix()
                _need(rel in expected_dirs and not _reparse(path.lstat()) and not path.is_symlink(), 'unexpected deployment directory')
            for child in files:
                path = Path(base)/child
                rel = path.relative_to(self.tree).as_posix()
                pin = remaining.pop(rel,None)
                _need(pin is not None and _hash_file(path,pin.size) == pin.sha256, 'deployment file changed or unexpected')
        _need(not remaining,'deployment files missing')
        return self


def stage_deployment(receipt, destination, *, max_expanded=MAX_EXPANDED):
    """Host-owned temporary staging that commits only after full verification."""
    _need(type(max_expanded) is int and 0 < max_expanded <= MAX_EXPANDED, 'invalid runtime extraction bound')
    verify_artifacts(receipt)
    dest = Path(destination).absolute()
    _need(dest.parent.resolve(strict=True) == dest.parent and not dest.exists() and not dest.is_symlink(),
          'deployment needs a fresh path below a trusted directory')
    lanes = {}
    for pin in receipt.artifacts:
        _need(WHEEL_NAME.fullmatch(pin.path) is not None, 'unexpected deployable wheel')
        lane,leaf = pin.path.split('/')
        _need(lane not in lanes, 'duplicate deployable wheel')
        lanes[lane] = (pin, '-'.join(leaf.split('-')[:2])+'.dist-info')
    _need(set(lanes) == {'engine','native'}, 'deployment requires both verified wheels')
    with TemporaryDirectory(dir=dest.parent,prefix='.level3-deploy-') as tmp:
        stage = Path(tmp)/'deploy'
        tree = stage/'site'
        tree.mkdir(parents=True)
        paths, entries, size = {}, [], 0
        for lane,(pin,dist) in sorted(lanes.items()):
            wheel = receipt.artifact_dir/pin.path
            with wheel.open('rb') as raw:
                hashed = sha256()
                while block := raw.read(BLOCK):
                    hashed.update(block)
                _need(hashed.hexdigest() == pin.sha256, 'wheel bytes changed before extraction')
                raw.seek(0)
                with ZipFile(raw) as z:
                    members = z.infolist()
                    _need(len(entries)+len(members) <= MAX_FILES, 'too many deployment files')
                    internal = {}
                    for member in members:
                        name = member.orig_filename
                        kind = 'directory' if name.endswith('/') else 'file'
                        has_mode = member.create_system == 3
                        mode = stat.S_IFMT(member.external_attr >> 16) if has_mode else 0
                        _need(member.compress_type in (ZIP_DEFLATED, ZIP_STORED) and not member.flag_bits & 1
                              and mode in (0,stat.S_IFDIR if kind == 'directory' else stat.S_IFREG), 'unsafe wheel member')
                        member_path = name[:-1] if kind == 'directory' else name
                        _claim(internal,member_path,kind)
                        if kind == 'directory':
                            continue
                        target = _export_root(member_path,lane=lane,dist_info=dist)
                        if target is None:
                            continue
                        _claim(paths,target,'file')
                        _need(size+member.file_size <= max_expanded, 'expanded wheel budget exhausted')
                        here = tree / target
                        here.parent.mkdir(parents=True,exist_ok=True)
                        count, h = 0, sha256()
                        with z.open(member) as inp, here.open('xb') as out:
                            while block := inp.read(BLOCK):
                                count += len(block)
                                _need(count <= member.file_size, 'wheel expanded beyond declared size')
                                out.write(block)
                                h.update(block)
                        _need(count == member.file_size,'truncated wheel member')
                        here.chmod(0o444)
                        size += count
                        entries.append(RuntimeFile(target,count,h.hexdigest()))
        required = {'sglang/__init__.py','sgl_kernel/__init__.py',*MAPS}
        _need(required <= {x.path for x in entries},'runtime missing required Engine or Native files')
        wheels = tuple(sorted(receipt.artifact_hashes.items()))
        deployment = Deployment(receipt.candidate_id,receipt.job_id,wheels,tuple(sorted(entries,key=lambda x:x.path)),stage)
        (stage/'manifest.json').write_bytes(canonical(deployment.record())+b'\n')
        _publish_bind_tree(tree)
        deployment.verify().verify_runtime_access()
        stage.rename(dest)
        return Deployment(deployment.candidate,deployment.build_job,deployment.wheels,deployment.files,dest)


def stage_component_deployment(parent, candidate, job_id, destination, modules, updates, deleted):
    """Compose a new immutable deployment without changing a byte of an older version."""
    parent.verify()
    destination = Path(destination).absolute()
    _need(destination.parent.resolve(strict=True) == destination.parent and not destination.exists()
          and not destination.is_symlink(), 'component destination must be a fresh trusted path')
    parent_files = {f.path: f for f in parent.files}
    _need(set(deleted) <= parent_files.keys() and not set(deleted) & updates.keys(), 'invalid component deletions')
    files = {path: item for path, item in parent_files.items() if path not in deleted}
    files.update(updates)
    _need({'sglang/__init__.py', 'sgl_kernel/__init__.py', *MAPS} <= files.keys(), 'missing core runtime files')
    _need(len(files) <= MAX_FILES and sum(f.size for f in files.values()) <= MAX_EXPANDED, 'component deployment capacity')
    seen = {}
    with TemporaryDirectory(dir=destination.parent, prefix='.level3-deploy-') as tmp:
        stage = Path(tmp)/'deploy'
        site = stage/'site'
        site.mkdir(parents=True)
        for name, item in sorted(files.items()):
            _claim(seen, name, 'file')
            target = site/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in updates:
                source = modules/name
                _need(_hash_file(source, item.size) == item.sha256, 'component output changed')
                source.chmod(0o444)
            else:
                source = parent.tree/name
            os.link(source, target)
        ordered = tuple(files[p] for p in sorted(files))
        deploy = Deployment(candidate, job_id, parent.wheels, ordered, stage, parent.identity)
        (stage/'manifest.json').write_bytes(canonical(deploy.record())+b'\n')
        _publish_bind_tree(site)
        deploy.verify().verify_runtime_access()
        stage.rename(destination)
    return Deployment(candidate, job_id, parent.wheels, ordered, destination, parent.identity)


def runtime_container_args(deployment, *, image, name, gpu_uuid=None, scratch_mib=SCRATCH_MIB, memory_mib=65536):
    """The controller, not the Agent, supplies the image, GPU and fixed launcher."""
    _need(type(image) is str and (re.fullmatch(r'sha256:[0-9a-f]{64}',image) or
          re.fullmatch(r'[a-z0-9][a-z0-9._/:-]+@sha256:[0-9a-f]{64}',image)), 'runtime image not content pinned')
    _need(type(name) is str and re.fullmatch(r'ee-l3-[a-z0-9-]{1,56}',name), 'invalid private container name')
    _need(all(type(x) is int and 16 <= x <= 131072 for x in (scratch_mib,memory_mib)), 'invalid runtime bounds')
    _need(',' not in str(deployment.tree), 'unsafe bind directory')
    deployment.verify_runtime_access()
    args = ['--name',name,'--network','none','--read-only','--user','12345:12345','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','512','--memory',f'{memory_mib}m',
            '--shm-size','2g',
            *(arg for path,opts in runtime_tmpfs(scratch_mib).items() for arg in ('--tmpfs',f'{path}:{opts}')),
            '--mount',f'type=bind,src={deployment.tree},dst={TARGET},readonly',
            *(arg for k,v in RUNTIME_ENV.items() for arg in ('--env',f'{k}={v}')),
            '--workdir','/tmp']
    if gpu_uuid is not None:
        _need(type(gpu_uuid) is str and re.fullmatch(r'GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',gpu_uuid,re.I),
              'GPU scope requires one pinned device UUID')
        args += ['--gpus',f'device={gpu_uuid}']
    return args


@dataclass(frozen=True)
class ServiceSpec:
    image: str
    image_id: str
    model_dir: Path
    model_files: tuple[tuple[str, str], ...]
    gguf: Path
    gguf_sha256: str
    gpu_uuid: str
    launch_profile: str = 'diagnostic_eager'

    def recheck_model(self, *, deadline=None):
        model = self.model_dir.resolve(strict=True)
        gguf = self.gguf.resolve(strict=True)
        _need(model == self.model_dir and model.is_dir() and gguf == self.gguf and gguf.is_file()
              and ',' not in str(model) and ',' not in str(gguf), 'invalid trusted model asset path')
        files = dict(self.model_files)
        _need(len(files) == len(self.model_files) and {'tokenizer.json','chat_template.jinja'} <= files.keys(),
              'tokenizer/template pins missing')
        _need(WEIGHT not in files,'GGUF asset must be pinned separately')
        seen = {}
        for name, expected in self.model_files:
            _claim(seen,name,'file')
            path = model/name
            _need(path.resolve(strict=True) == path and re.fullmatch(r'[0-9a-f]{64}',expected) and
                  _hash_file(path,path.stat().st_size,deadline=deadline) == expected,'trusted model asset hash mismatch')
        # Mount neither unregistered tokenizer assets nor model-side executables.
        staged = {path.relative_to(model).as_posix() for path in model.rglob('*') if path.is_file() or path.is_symlink()}
        # A regular empty mount point permits Docker to overlay the separately
        # pinned GGUF file even when the model-directory bind is read-only.
        if WEIGHT in staged:
            placeholder = model/WEIGHT
            _need(not placeholder.is_symlink() and placeholder.is_file() and placeholder.stat().st_size == 0,
                  'invalid GGUF bind mount point in model directory')
            staged.remove(WEIGHT)
        _need(staged == files.keys(), 'model directory contains unregistered bytes')
        _need(re.fullmatch(r'[0-9a-f]{64}',self.gguf_sha256) and
              _hash_file(gguf,gguf.stat().st_size,deadline=deadline) == self.gguf_sha256,
              'GGUF artifact hash mismatch')
        return {**files,WEIGHT:self.gguf_sha256}


WEIGHT = 'Spark-X2.6-1.7B-Q8_0.gguf'
LAUNCH = ('-m','sglang.launch_server','--model-path',f'/model/{WEIGHT}','--load-format','gguf',
          '--tokenizer-path','/model','--chat-template','/model/chat_template.jinja','--trust-remote-code',
          '--host','127.0.0.1','--port','30091','--served-model-name','spark26-q8','--tp-size','1',
          '--context-length','8192','--max-total-tokens','16384','--mem-fraction-static','0.4',
          '--max-running-requests','4','--attention-backend','triton','--disable-radix-cache','--stream-interval','1',
          '--disable-decode-cuda-graph','--disable-prefill-cuda-graph','--tool-call-parser','spark25',
          '--reasoning-parser','qwen3')


IMMEDIATE_GRAPH_WIRE = 'graph_defaults_incremental'
GRAPH_PROFILES = {
    'diagnostic_eager': frozenset(),
    'decode_graph': frozenset({'--disable-decode-cuda-graph'}),
    'graph_defaults': frozenset({'--disable-decode-cuda-graph', '--disable-prefill-cuda-graph'}),
    IMMEDIATE_GRAPH_WIRE: frozenset({'--disable-decode-cuda-graph', '--disable-prefill-cuda-graph'}),
}


def launch_args(spec, deployment=None):
    from .v3_level3_startup import apply
    _need(spec.launch_profile in GRAPH_PROFILES, 'unregistered launch profile')
    allowed_graphs = GRAPH_PROFILES[spec.launch_profile]
    args = tuple(arg for arg in LAUNCH if arg not in allowed_graphs)
    if spec.launch_profile == IMMEDIATE_GRAPH_WIRE:
        # The pinned E0 cumulative path can hand the serializer a borrowed
        # output_ids list which grows after completion_tokens was captured.
        # Emit disjoint snapshots in both arms; keep the verifier fail-closed.
        args = (*args, '--incremental-streaming-output')
    return apply(args, getattr(deployment,'runtime_config',b''))


def validate_container(raw, deployment, spec, name):
    """Inspect from the trusted Docker daemon before candidate code can run."""
    host, conf = raw['HostConfig'],raw['Config']
    mounts = {m['Destination']:(m['Source'],m['RW'],m['Type']) for m in raw['Mounts']}
    pins = {TARGET:(str(deployment.tree),False,'bind'), '/model':(str(spec.model_dir),False,'bind'),
            f'/model/{WEIGHT}':(str(spec.gguf),False,'bind')}
    env = dict(line.split('=',1) for line in conf['Env'] if '=' in line)
    request, = host['DeviceRequests']
    _need(raw['Name'] == '/'+name and raw['Image'] == spec.image_id and not raw['State']['Running']
          and host['NetworkMode'] == 'none' and host['ReadonlyRootfs'] is True and host['Privileged'] is False
          and host['PidMode'] in ('','private') and host['IpcMode'] == 'private' and not host['PortBindings']
          and not host['Devices'] and host['CapDrop'] == ['ALL'] and not host['CapAdd']
          and host['SecurityOpt'] == ['no-new-privileges'] and host['Tmpfs'] == runtime_tmpfs()
          and host['PidsLimit'] == 512 and host['Memory'] == 65536 * 1024**2
          and request['DeviceIDs'] == [spec.gpu_uuid] and request['Count'] == 0
          and ['gpu'] in request['Capabilities'] and conf['User'] == '12345:12345'
          and conf['Entrypoint'] == ['python'] and conf['Cmd'] == list(launch_args(spec,deployment))
          and all(env.get(k)==v for k,v in RUNTIME_ENV.items()) and mounts == pins,
          'candidate container does not match pinned runtime')
    return raw['Id']


@dataclass(frozen=True)
class StartedService:
    name: str
    container_id: str
    deployment: Deployment
    inspection: dict
    model_hashes: dict
    weight_check_started_ns: int
    weight_check_finished_ns: int


class DockerService:
    """Launches only the frozen candidate service. Caller must hold the node lease."""
    def __init__(self, spec, *, docker='/usr/bin/docker', diagnostics=None):
        self.spec, self.docker = spec,docker
        self.diagnostics = None if diagnostics is None else Path(diagnostics)
        if self.diagnostics is not None:
            self.diagnostics.mkdir(mode=0o700)

    def _retain_diagnostics(self, name, deadline):
        budget = min(3,deadline-monotonic()-25)
        if self.diagnostics is None or budget <= 0:
            return
        try:
            captured = self._call(('logs','--tail','160',name),budget)
            with (self.diagnostics/(name+'.log')).open('xb') as out:
                # Container uses a size-limited local log driver. Never share
                # untrusted service log bytes with the Agent or verdict parser.
                out.write((captured.stdout+captured.stderr)[-128*1024:])
        except (OSError,subprocess.SubprocessError):
            pass  # diagnostics cannot take precedence over verified removal


    def _call(self,args,timeout):
        return subprocess.run(docker_command(self.docker,*args),stdin=subprocess.DEVNULL,capture_output=True,
                              timeout=timeout,check=False,env=docker_env())

    @contextmanager
    def start(self, deployment, name, deadline):
        spec = self.spec
        args = runtime_container_args(deployment,image=spec.image,name=name,gpu_uuid=spec.gpu_uuid)
        deployment.verify()
        weight_check_started_ns = monotonic_ns()
        weights = spec.recheck_model(deadline=deadline)
        weight_check_finished_ns = monotonic_ns()
        def run(*argv):
            limit = min(25,deadline-monotonic())
            _need(limit > 0, 'remaining service lease exhausted')
            result = self._call(argv,limit)
            _need(result.returncode == 0,'Docker service command failed')
            return result.stdout
        image = run('image','inspect','--format','{{.Id}}',spec.image).decode().strip()
        _need(image == spec.image_id,'runtime image identity drift')
        _need(self._call(('inspect',name),5).returncode != 0,'candidate name already in use')
        cleanup_needed = False
        try:
            cleanup_needed = True
            raw_id = run('create',*args,'--pull=never','--ipc','private','--no-healthcheck','--restart','no',
                         '--log-driver','local','--log-opt','max-size=8m',
                         '--log-opt','max-file=2','--entrypoint','python',
                         '--mount',f'type=bind,src={spec.model_dir},dst=/model,readonly',
                         '--mount',f'type=bind,src={spec.gguf},dst=/model/{WEIGHT},readonly',spec.image,*launch_args(spec,deployment))
            output = json.loads(run('inspect',name))
            _need(type(output) is list and len(output)==1,'ambiguous candidate container')
            inspected = output[0]
            container_id = validate_container(inspected,deployment,spec,name)
            _need(re.fullmatch(r'[0-9a-f]{64}',container_id) and raw_id.decode().strip() == container_id,
                  'created candidate identity drift')
            run('start',container_id)
            yield StartedService(name,container_id,deployment,inspected,weights,
                                 weight_check_started_ns,weight_check_finished_ns)
        finally:
            if cleanup_needed:
                try:
                    self._retain_diagnostics(name,deadline)
                finally:
                    self._remove(name)

    def _remove(self, name):
        try:
            self._call(('rm','-f',name),15)
            found = self._call(('inspect',name),8)
            _need(found.returncode == 1 and b'no such' in found.stderr.lower(),'container removal not verified')
        except (BuildError,OSError,subprocess.SubprocessError) as exc:
            raise CleanupUncertain('candidate service cleanup not verified') from exc
