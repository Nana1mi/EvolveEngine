"""Linux host identity, exclusive device lease, and external mapping observations."""
from contextlib import contextmanager
from dataclasses import asdict
import mmap
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
from tempfile import NamedTemporaryFile
from time import monotonic, monotonic_ns

from .v3_level3_build import CleanupUncertain, DOCKER_IMAGE_PLATFORM_AND_VOLUMES
from .v3_level3_evidence import hash_linux_mapping, observe_loaded_mappings, parse_nvidia_smi_process_memory

GPU = re.compile(r'GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\Z',re.I)
NODE = 'spark-c3bc'
EXPERIMENT_USER = 'jiazheng'
REGISTERED_GPU_LOCK = Path('/home/jiazheng/.evolve-spark26-gpu.lock')
DOCKER_SOCKET = 'unix:///var/run/docker.sock'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def docker_command(binary, *args):
    return [binary,'--host',DOCKER_SOCKET,*args]


def docker_env():
    return {'PATH':'/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','LC_ALL':'C',
            'DOCKER_CONFIG':'/dev/null'}


def command(argv, deadline, *, env=None):
    left = min(15,deadline-monotonic())
    need(left > 0,'node deadline exhausted')
    result = subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,check=False,timeout=left,env=env)
    need(result.returncode == 0 and len(result.stdout) <= 128*1024, 'host command failed or output exceeded bound')
    return result.stdout.decode('utf-8')


class Gb10:
    """Configure with the existing shared experiment lock, not a new parallel lock."""
    def __init__(self, gpu_uuid, lock, *, docker='/usr/bin/docker', nvidia_smi='/usr/bin/nvidia-smi'):
        need(type(gpu_uuid) is str and GPU.fullmatch(gpu_uuid),'need pinned GPU UUID')
        self.gpu_uuid,self.lock,self.docker,self.smi = gpu_uuid,Path(lock),docker,nvidia_smi

    def compute(self, deadline):
        raw = command([self.smi,'-i',self.gpu_uuid,'--query-compute-apps=pid','--format=csv,noheader'],deadline)
        pids = raw.split()
        need(all(re.fullmatch('[1-9][0-9]*',p) for p in pids),'GPU occupancy cannot be verified')
        return [int(p) for p in pids]

    @classmethod
    def discover(cls, deadline, *, docker='/usr/bin/docker', nvidia_smi='/usr/bin/nvidia-smi'):
        need(platform.system() == 'Linux' and platform.machine() in ('aarch64','arm64') and socket.gethostname() == NODE,
             'not the registered ARM64 development node')
        raw = command([nvidia_smi,'--query-gpu=uuid','--format=csv,noheader'],deadline)
        ids = raw.splitlines()
        need(len(ids)==1 and GPU.fullmatch(ids[0].strip()),'expected exactly one GB10 device')
        return cls(ids[0].strip(),REGISTERED_GPU_LOCK,docker=docker,nvidia_smi=nvidia_smi)

    def inventory(self, deadline):
        need(platform.system() == 'Linux' and platform.machine() in ('aarch64','arm64') and socket.gethostname() == NODE,
             'not the registered ARM64 development node')
        raw = command([self.smi,'-i',self.gpu_uuid,'--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],deadline)
        rows = raw.strip().splitlines()
        need(len(rows)==1,'registered device is ambiguous')
        uuid,name,driver = (s.strip() for s in rows[0].split(','))
        need(uuid == self.gpu_uuid and 'GB10' in name and bool(driver),'not the registered GB10 device')
        daemon = command(docker_command(self.docker,'info','--format','{{.Architecture}} {{.ServerVersion}}'),
                         deadline,env=docker_env()).strip()
        fields = daemon.split()
        need(len(fields)==2 and fields[0] in ('aarch64','arm64'),'not a local ARM64 Docker daemon')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return dict(worker_id=NODE,boot_id=boot,device_uuid=uuid,device_name=name,driver=driver,
                    docker_info=daemon,compute_pids=self.compute(deadline))

    @property
    def quarantine(self):
        return self.lock.with_name('.evolve-level3-cleanup-uncertain')

    def leftover_jobs(self, deadline):
        raw = command(docker_command(self.docker,'ps','-a','--format','{{.Names}}'),deadline,env=docker_env())
        return [name for name in raw.splitlines() if name.startswith(('ee-native-','ee-l3-job-','ee-l3-observer-'))]

    def mark_quarantine(self):
        try:
            fd = os.open(self.quarantine,os.O_CLOEXEC|os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        except FileExistsError:
            return
        try:
            os.write(fd,b'Level 3 cleanup must be verified by a trusted operator before another job.\n')
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def registered_lease(self):
        import fcntl
        import pwd
        need(platform.system() == 'Linux' and socket.gethostname() == NODE and self.lock == REGISTERED_GPU_LOCK
             and self.lock.parent.resolve(strict=True)==self.lock.parent,
             'GPU lease does not target the registered worker lock')
        owner = pwd.getpwnam(EXPERIMENT_USER).pw_uid
        # A privileged observer must use the existing experiment-account inode.
        # Missing or replaced locks are errors; do not create a parallel lease.
        fd = os.open(self.lock,os.O_CLOEXEC|os.O_NOFOLLOW|os.O_RDWR|os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid in (owner,0),
                 'untrusted GPU lock metadata')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            need(os.path.samestat(info,self.lock.lstat()), 'registered GPU lock inode changed')
            yield
        finally:
            os.close(fd)

    def check_registered_lease(self):
        with self.registered_lease():
            return True

    @contextmanager
    def exclusive(self, deadline):
        with self.registered_lease():
            need(not self.quarantine.exists() and not self.quarantine.is_symlink(),'node cleanup quarantine is active')
            need(not self.inventory(deadline)['compute_pids'],'registered device is occupied')
            need(not self.leftover_jobs(deadline),'leftover Level 3 objects require operator cleanup')
            try:
                yield
            except CleanupUncertain:
                self.mark_quarantine()
                raise
            finally:
                try:
                    until = min(deadline,monotonic()+10)
                    if self.compute(until) or self.leftover_jobs(until):
                        raise CleanupUncertain('job cleanup not verified; do not kill unknown processes')
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    self.mark_quarantine()
                    raise CleanupUncertain('could not recheck post-job resources') from exc
                except CleanupUncertain:
                    self.mark_quarantine()
                    raise

    def memory(self, deadline):
        return command([self.smi,'-i',self.gpu_uuid,'--query-gpu=uuid,memory.used',
                        '--format=csv,noheader,nounits'],deadline)

    def compute_memory(self, deadline):
        return command([self.smi,'-i',self.gpu_uuid,'--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'],deadline)

    def image(self, locator, deadline):
        need(type(locator) is str and re.fullmatch(r'(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}',locator),
             'image is not content-addressed')
        raw = command(docker_command(self.docker,'image','inspect','--format',
                       '{{.Id}} '+DOCKER_IMAGE_PLATFORM_AND_VOLUMES,locator),deadline,env=docker_env())
        fields = raw.strip().split(maxsplit=2)
        need(len(fields)==3 and re.fullmatch(r'sha256:[0-9a-f]{64}',fields[0])
             and fields[1]=='linux/arm64' and fields[2] in ('null','{}'),'unusable pinned ARM64 image')
        return fields[0]

    def pids(self, container_id, deadline):
        need(type(container_id) is str and re.fullmatch('[0-9a-f]{64}',container_id),'bad service identity')
        raw = command(docker_command(self.docker,'top',container_id,'-eo','pid'),deadline,env=docker_env())
        head,*rows = raw.splitlines()
        need(head.strip()=='PID' and 0 < len(rows) <= 512
             and all(re.fullmatch('[1-9][0-9]*',p.strip()) for p in rows),'invalid Docker process inventory')
        return {int(p) for p in rows}

    def service_memory(self, service, deadline):
        before = {pid:process_identity(Path('/proc')/str(pid)) for pid in self.pids(service.container_id,deadline)}
        raw = self.compute_memory(deadline)
        t = monotonic_ns()
        parsed = parse_nvidia_smi_process_memory(raw,self.gpu_uuid)
        seen = set(parsed)
        need(seen and seen <= before.keys() and seen <= self.pids(service.container_id,deadline),
             'no exclusively owned service compute-memory samples')
        after = {pid:process_identity(Path('/proc')/str(pid)) for pid in seen}
        need(all(before[pid]==after[pid] for pid in seen),'compute-service PID identity changed during observation')
        return dict(at_ns=t,raw_nvml=raw,service_processes={str(pid):data for pid,data in after.items()},
                    used_bytes=sum(parsed.values()))

    def mappings(self, service, variant, deadline):
        paths,_ = service.deployment.native_mapping(variant)
        start_ns = monotonic_ns()
        observations = []
        pids = self.pids(service.container_id,deadline)
        for pid in sorted(pids):
            need(monotonic()<deadline,'mapping observation exceeded lease')
            proc = Path('/proc')/str(pid)
            try:
                identity = process_identity(proc)
                with (proc/'maps').open(encoding='utf8') as scan:
                    raw = scan.read(8*1024*1024+1)
                need(len(raw)<=8*1024*1024,'process maps exceeds observer bound')
                if not any(line.split(maxsplit=5)[-1] in paths.values() for line in raw.splitlines() if line):
                    continue
                backing = {}
                def read(address):
                    item = hash_linux_mapping(proc,address)
                    backing[address] = asdict(item)
                    return item
                observed = observe_loaded_mappings(raw,paths,read)
                need(process_identity(proc)==identity,'service PID identity changed during observation')
            except FileNotFoundError:
                continue
            observations.append(dict(pid=pid,process=identity,maps=raw,backing=backing,loaded=observed))
        need(observations and {o['pid'] for o in observations}<=self.pids(service.container_id,deadline),
             'no stable candidate process mapping was witnessed')
        first = observations[0]['loaded']
        need(all(o['loaded']==first for o in observations),'inconsistent loaded Native mappings')
        return dict(container=service.container_id,loaded=first,observations=observations,
                    start_ns=start_ns,end_ns=monotonic_ns())


def process_identity(proc):
    raw = (proc/'stat').read_text()
    suffix = raw.rsplit(')',1)[1].split()
    need(len(suffix)>19 and suffix[19].isdigit(),'malformed process identity')
    return dict(start_ticks=int(suffix[19]),mount_ns=(proc/'ns/mnt').stat().st_ino)


def check_mapping_read():
    """Preflight only: a local memmap cannot establish candidate mapped identity."""
    from hashlib import sha256
    payload = b'Level 3 host map_files preflight\n'
    with NamedTemporaryFile() as temp:
        temp.write(payload)
        temp.flush()
        with mmap.mmap(temp.fileno(),len(payload),access=mmap.ACCESS_READ):
            proc = Path('/proc')/str(os.getpid())
            matches = [line.split(maxsplit=5)[0] for line in (proc/'maps').read_text().splitlines()
                       if line.split(maxsplit=5)[-1]==temp.name]
            need(len(matches)==1,'test process mapping absent')
            found = hash_linux_mapping(proc,matches[0])
            need(found.sha256==sha256(payload).hexdigest(),'mapped file bytes differ')
    return True
