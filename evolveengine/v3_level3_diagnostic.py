"""Non-scoring, isolated CUDA diagnostics for a verified Native deployment.

This is the trusted worker's spot runner, not a network service. It cannot
sign correctness or speed, change the Engine, or access the Judge.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
from secrets import token_hex
import subprocess
from time import monotonic

from .v3_harness import canonical, digest
from .v3_level3_collect import DockerObserver, artifact_space, stage_inputs
from .v3_level3_evidence import file_sha256
from .v3_level3_deploy import DockerService, RUNTIME_ENV, TARGET, runtime_container_args, runtime_tmpfs
from .v3_level3_gb10 import docker_command, docker_env
from .v3_level3_worker import ObservationUnavailable

SCHEMA = 'level3-private-diagnostic-v1'
MOUNT = '/ee-diagnostic'
ENTRY = 'python'
ARGV = ('-B','-s',MOUNT+'/probe.py')
MAX_SCRIPT = 24 << 10
ENGINEERING_SCRIPT = 2 << 20
MAX_OUTPUT = 32 << 10
MIN_MS = 20000
MAX_MS = 180000
ENGINEERING_MS = 2 * 60 * 60 * 1000
MEMORY_MIB = 8192
SCRATCH_MIB = 1024


def need(ok, why):
    if not ok:
        raise ValueError(why)


def document(path, data):
    raw = canonical(data)+b'\n'
    with path.open('xb') as out:
        out.write(raw)
    return digest(raw)


def request(source, *, max_script=MAX_SCRIPT):
    need(type(source) is str,'diagnostic source must be UTF-8 text')
    raw = source.encode('utf-8')
    need(0<len(raw)<=max_script and b'\x00' not in raw,'invalid diagnostic input size or encoding')
    # Syntax verification never evaluates the diagnostic in the trusted host.
    try:
        compile(raw,'<diagnostic>','exec')
    except (SyntaxError,ValueError,RecursionError):
        raise ValueError('diagnostic is not a single Python source file') from None
    return raw


def attest(raw, deployment, image, gpu, inputs, name):
    host, config = raw['HostConfig'],raw['Config']
    devices = host['DeviceRequests']
    keys = {m['Destination']:(m['Source'],m['RW'],m['Type']) for m in raw['Mounts']}
    env = dict(item.split('=',1) for item in config['Env'])
    ident = raw['Id']
    need(re.fullmatch('[a-f0-9]{64}',ident) and raw['Name']=='/'+name and raw['Image']==image
         and raw['State']['Running'] is False and host['NetworkMode']=='none' and host['ReadonlyRootfs'] is True
         and host['Privileged'] is False and host['PidMode'] in ('','private') and host['IpcMode']=='private'
         and not host['PortBindings'] and not host['Devices'] and host['CapDrop']==['ALL'] and not host['CapAdd']
         and host['SecurityOpt']==['no-new-privileges'] and host['PidsLimit']==512
         and host['Memory']==MEMORY_MIB*(1<<20) and host['Tmpfs']==runtime_tmpfs(SCRATCH_MIB)
         and len(devices)==1 and devices[0]['DeviceIDs']==[gpu] and devices[0]['Count']==0
         and ['gpu'] in devices[0]['Capabilities'] and config['User']=='12345:12345'
         and config['Entrypoint']==[ENTRY] and config['Cmd']==list(ARGV)
         and all(env.get(k)==v for k,v in RUNTIME_ENV.items())
         and len(raw['Mounts'])==len(keys) and keys=={TARGET:(str(deployment.tree),False,'bind'),MOUNT:(str(inputs),False,'bind')},
         'diagnostic runtime scope drift')
    return ident


@dataclass(frozen=True)
class DiagnosticResult:
    engine: str
    build: str
    script: str
    status: str
    node_ms: int
    evidence: str
    stdout_bytes: int
    stdout_sha256: str | None
    stderr_sha256: str | None


class DockerDiagnostic(DockerService):
    def __init__(self, spec, *, docker, exclusive, retain_artifacts=False, extended=False):
        super().__init__(spec,docker=docker)
        self.exclusive = exclusive
        self.retain_artifacts = retain_artifacts
        self.maximum_ms = ENGINEERING_MS if extended else MAX_MS
        self.maximum_script = ENGINEERING_SCRIPT if extended else MAX_SCRIPT

    def launch(self, deployment, build, source, root, cap_ms):
        need(type(cap_ms) is int and MIN_MS<=cap_ms<=self.maximum_ms,'diagnostic lease outside registered limits')
        need(type(build) is str and re.fullmatch('[a-f0-9]{64}',build),'missing verified build identity')
        raw = request(source, max_script=self.maximum_script)
        script = digest(raw)
        began = monotonic()
        deadline = began+cap_ms/1000
        output = Path(root).resolve(strict=True)/('diag-'+token_hex(10))
        output.mkdir(mode=0o700)
        ident = dict(schema=SCHEMA,engine=deployment.candidate,deployment=deployment.identity,build=build,script=script,
                     max_node_ms=cap_ms,scoring=False,weight_mounts=False)
        document(output/'reservation.json',ident)
        inputs = output/'input'
        stage_inputs(inputs,{'probe.py':raw})
        name = 'ee-l3-job-'+output.name

        def dock(*argv):
            left = min(15,deadline-monotonic()-13)
            need(left>0,'diagnostic has no cleanup margin')
            proc = self._call(argv,left)
            need(proc.returncode==0 and len(proc.stdout)<=1<<20,'diagnostic Docker control failure')
            return proc.stdout

        status = 'completed_non_scoring'
        try:
            with self.exclusive(deadline):
                deployment.verify()
                need(dock('image','inspect','--format','{{.Id}}',self.spec.image).decode().strip()==self.spec.image_id,
                     'diagnostic runtime image identity changed')
                need(',' not in str(inputs),'unsafe diagnostic bind source')
                missing = self._call(('inspect',name),3)
                need(missing.returncode==1 and b'no such' in missing.stderr.lower(),'diagnostic container is not fresh')
                args = runtime_container_args(deployment,image=self.spec.image,name=name,gpu_uuid=self.spec.gpu_uuid,
                                              scratch_mib=SCRATCH_MIB,memory_mib=MEMORY_MIB)
                try:
                    cid = dock('create',*args,'--pull=never','--ipc','private','--no-healthcheck','--restart','no',
                               '--log-driver','none','--mount',f'type=bind,src={inputs},dst={MOUNT},readonly',
                               '--entrypoint',ENTRY,self.spec.image,*ARGV).decode().strip()
                    view, = json.loads(dock('inspect',name))
                    need(cid==attest(view,deployment,self.spec.image_id,self.spec.gpu_uuid,inputs,name),
                         'unexpected diagnostic container identity')
                    document(output/'docker.inspect.json',view)
                    try:
                        allowance = artifact_space(output) if self.retain_artifacts else MAX_OUTPUT
                        DockerObserver.capture(docker_command(self.docker,'start','-a',cid),output,deadline-15,
                                               allowance,env=docker_env(),artifact=self.retain_artifacts)
                    except (ObservationUnavailable, subprocess.SubprocessError) as exc:
                        status = 'incomplete_non_scoring'
                        document(output/'diagnostic-error.json',dict(error=type(exc).__name__,reason=str(exc)[:200]))
                finally:
                    # Always remove the entire job, not merely the attach client.
                    self._remove(name)
        except (Exception,KeyboardInterrupt) as exc:
            document(output/'unsettled.json',dict(error=type(exc).__name__,reservation_retained_ms=cap_ms))
            raise
        ms = int((monotonic()-began)*1000)
        if ms>cap_ms:
            document(output/'unsettled.json',dict(error='node_lease_overrun',observed_ms=ms,reservation_retained_ms=cap_ms))
            raise TimeoutError('diagnostic node lease overrun')
        stdout,stderr = output/'observer.stdout',output/'observer.stderr'
        size = stdout.stat().st_size if stdout.is_file() else 0
        fields=dict(engine=deployment.candidate,build=build,script=script,status=status,node_ms=ms,stdout_bytes=size,
                    stdout_sha256=file_sha256(stdout) if size else None,
                    stderr_sha256=file_sha256(stderr) if stderr.is_file() and stderr.stat().st_size else None)
        evidence = document(output/'result.json',dict(**fields,kind=SCHEMA,non_scoring=True,verified_removed=True))
        return DiagnosticResult(evidence=evidence,**fields),output
