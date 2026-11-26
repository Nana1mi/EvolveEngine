"""One isolated Pi session shell for fixed/evolving/evaluation modes.

The disposable guest has no network and sees only per-launch RO inputs and a private
socket. Provider, policy, and development callbacks run on the trusted host. No shell
commands from an Agent are executed on the host.
"""
import base64
from dataclasses import dataclass
import hmac
import json
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from threading import Event, Lock, Thread
import time
from collections import deque

from .v3_harness import canonical, digest, relative
from .v3_provider import LeaseError


HERE=Path(__file__).resolve().parent
MAX_FRAME=768*1024
MAX_REPLY=4*1024*1024
LONG_MODEL_REQUEST=16*1024*1024
LONG_MODEL_OUTPUT=393216
RUNNER_FILES=(HERE/'v3_guest.mjs',HERE/'v3_rpc.mjs',HERE/'v3_relay.py',HERE/'v3_paged_read.mjs',Path(__file__))
IMAGE=re.compile(r'sha256:[0-9a-f]{64}\Z')


def need(condition,message):
    if not condition:raise ValueError(message)


def docker(*args,timeout=20,check=True):
    p=subprocess.run(['docker',*map(str,args)],capture_output=True,timeout=timeout)
    if check and p.returncode:raise RuntimeError(f'Docker {args[0]} failed (rc={p.returncode})')
    return p


def ro(src,dst,kind='bind',writable=False):
    return f'type={kind},src={src},dst={dst}'+('' if writable else ',readonly')


def launch(name):
    return ['--name',name,'--network','none','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','160','--memory','768m','--cpus','1']


def exception_details(error):
    # Stack frames only: exception messages, source text, locals and arguments can
    # contain API keys, prompts, private paths, or adversarial tool data.
    stack = deque(maxlen=24)
    frame = error.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        stack.append(dict(file=Path(code.co_filename).name, line=frame.tb_lineno, function=code.co_name))
        frame = frame.tb_next
    return dict(error=type(error).__name__,stack=list(stack))


class RpcAudit:
    """Trusted metadata-only JSONL, separate from Pi, score evidence and the dashboard."""
    def __init__(self, path):
        self.path, self.lock = Path(path), Lock()

    def __call__(self, **entry):
        with self.lock, self.path.open('ab') as stream:
            stream.write(canonical(dict(at_ns=time.time_ns(), **entry))+b'\n')


class Gateway:
    def __init__(self,provider,handler,token=None,*,delivery_complete=None,audit=None):
        self.provider=provider;self.handler=handler;self.token=token or secrets.token_urlsafe(26)
        self.events=[]
        self.delivery_complete=delivery_complete
        self.audit=audit
        self.active_tool=None
        self.sequence=0
        self.audit_failed=False

    def _audit(self, **record):
        if self.audit is not None:
            try:
                self.audit(**record)
            except Exception as exc:
                if not self.audit_failed:
                    self.audit_failed=True
                    self.events.append(dict(kind='audit_unavailable',error=type(exc).__name__))

    def _fault(self, route, exc, **fields):
        detail = exception_details(exc)
        # Diagnostics are not copied into segment reports, guest replies or the public trace.
        self.events.append(dict(kind='infrastructure_error',route=route,error=detail['error'],**fields))
        self._audit(event='infrastructure_error',route=route,**fields,**detail)

    def background_audit(self, **entry):
        self._audit(**entry)

    def stalled_tool(self, seconds):
        active=self.active_tool
        return bool(active and time.monotonic()-active[1] >= seconds and
                    not getattr(self.handler, 'worker_active', lambda: False)())

    def dispatch(self,path,raw,auth):
        if path=='/v1/chat/completions':
            if not isinstance(auth,str) or not hmac.compare_digest(auth,'Bearer '+self.token):
                self.events.append(dict(kind='denied',route='provider'));return 403,'application/json',b'{"denied":true}'
            try:
                if self.delivery_complete and self.delivery_complete():
                    self.events.append(dict(kind='stopped',route='provider',reason='agent_delivered'))
                    terminal = dict(model='deepseek-flash', choices=[dict(index=0,delta={},finish_reason='stop')])
                    return 200,'text/event-stream',b'data: '+canonical(terminal)+b'\n\ndata: [DONE]\n\n'
                body = self.provider.exchange(raw)
                need(type(body) is bytes, 'invalid model response type')
                return 200,'text/event-stream',body
            except (ValueError,LeaseError):
                self.events.append(dict(kind='denied',route='provider_protocol_or_budget'));return 400,'application/json',b'{"error":"model request unavailable"}'
            except Exception as exc:
                self._fault('provider',exc)
                return 500,'application/json',b'{"error":"model infrastructure error"}'
        if isinstance(path,str) and re.fullmatch(r'/rpc/[a-z_]{1,40}',path):
            return self._tool(path[5:],raw)
        self.events.append(dict(kind='denied',route='unregistered'));return 404,'application/json',b'{"denied":true}'

    def _tool(self, tool, raw):
        self.sequence+=1
        call=self.sequence
        start=time.monotonic()
        self.active_tool=(call,start)
        self._audit(event='tool_started',call=call,tool=tool)
        status=500
        try:
            if self.delivery_complete and self.delivery_complete():
                self.events.append(dict(kind='stopped',route='tool',reason='agent_delivered'))
                status=409
                return status,'application/json',b'{"error":"agent already delivered"}'
            try:
                data=json.loads(raw);need(type(data) is dict,'RPC requires an object')
                code, result=self.handler(tool,data)
            except (ValueError,KeyError,TypeError,PermissionError):
                self.events.append(dict(kind='denied',route='tool_request'))
                status=403
                return status,'application/json',b'{"denied":true}'
            body=canonical(result)
            if type(code) is not int or not 100 <= code <= 599 or len(body) > MAX_REPLY:
                raise RuntimeError('invalid tool response envelope')
            status=code
            return status,'application/json',body
        except Exception as exc:
            self._fault('tool',exc,call=call,tool=tool)
            return status,'application/json',b'{"error":"tool infrastructure error","status":"infrastructure_error"}'
        finally:
            self._audit(event='tool_finished',call=call,tool=tool,status=status,
                        duration_ms=round((time.monotonic()-start)*1000))
            self.active_tool=None


class RelayUnsettled(RuntimeError):
    pass


class Relay:
    """Bounded sequence-numbered stdio transport; sidecar gets no host network or keys."""
    def __init__(self,bridge_name,image,volume,gateway):
        self.ready=Event();self.dispatch=gateway.dispatch;self.error=None;self.count=0
        self.gateway=gateway
        limits = getattr(gateway.provider, 'limits', None)
        self.frame_limit = MAX_FRAME
        self.model_reply_limit = MAX_REPLY
        self.binary_replies = False
        self.model_socket_idle_seconds = getattr(gateway.provider, 'model_socket_idle_seconds', 390)
        need(type(self.model_socket_idle_seconds) is int and 1 <= self.model_socket_idle_seconds <= 1800,
             'invalid model socket idle timeout')
        relay_options = []
        if limits is not None and (limits.per_turn_output > 8192 or hasattr(gateway.provider, 'model_socket_idle_seconds')):
            need(limits.per_turn_output <= LONG_MODEL_OUTPUT and limits.request_bytes <= LONG_MODEL_REQUEST,
                 'unsupported long model relay caps')
            self.frame_limit = max(MAX_FRAME, 4*((limits.request_bytes+2)//3)+4096)
            self.model_reply_limit = gateway.provider.sse_byte_limit(limits.per_turn_output)
            self.binary_replies = True
            relay_options = [str(limits.request_bytes), str(self.model_reply_limit), str(self.model_socket_idle_seconds)]
        args=['docker','run','--rm','-i',*launch(bridge_name),'--read-only','--mount',ro(volume,'/gate','volume',True),
              '--mount',ro(HERE/'v3_relay.py','/opt/relay.py'),image,'python','-B','-u','/opt/relay.py',*relay_options]
        self.err=tempfile.TemporaryFile();self.proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err)
        self.thread=Thread(target=self._serve,daemon=True);self.thread.start()

    def _serve(self):
        try:
            need(json.loads(self.proc.stdout.readline(4096))=={'ready':True},'relay failed startup')
            self.ready.set()
            while line:=self.proc.stdout.readline(self.frame_limit+1):
                need(line.endswith(b'\n') and len(line)<=self.frame_limit,'oversize relay frame')
                f=json.loads(line);need(set(f)=={'id','path','body','auth'} and type(f['id']) is int and f['id']==self.count+1,'invalid relay envelope')
                raw=base64.b64decode(f['body'],validate=True)
                status,typ,body=self.dispatch(f['path'],raw,f['auth'])
                reply_limit = self.model_reply_limit if status == 200 and typ == 'text/event-stream' else MAX_REPLY
                need(type(body) is bytes and len(body)<=reply_limit,'reply exceeds size cap')
                reply = dict(id=f['id'], status=status, type=typ)
                if self.binary_replies:
                    # Relay the verified body without building another base64 copy.
                    self.proc.stdin.write(canonical(dict(reply, length=len(body))) + b'\n')
                    self.proc.stdin.write(body)
                else:
                    self.proc.stdin.write(canonical(dict(reply, body=base64.b64encode(body).decode('ascii'))) + b'\n')
                self.proc.stdin.flush()
                self.count += 1
        except Exception as exc:
            self.error=type(exc).__name__
            self.ready.set()
            self.gateway._fault('relay',exc)

    def wait(self):
        need(self.ready.wait(22) and self.proc.poll() is None and not self.error,'relay unavailable')

    def close(self):
        try:self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
        self.thread.join(8)
        if self.thread.is_alive():
            # A Python callback cannot safely be killed. Do not block closing a
            # pipe it may still own, and never start a replacement segment.
            raise RelayUnsettled('trusted operation did not settle')
        self.proc.stdin.close();self.proc.stdout.close();self.err.close()


@dataclass(frozen=True)
class Images:
    pi: str
    bridge: str
    def verify(self):
        for im in (self.pi,self.bridge):
            need(isinstance(im,str) and IMAGE.fullmatch(im),'mutable image reference denied')
            need(docker('image','inspect',im,'--format','{{.Id}}').stdout.strip().decode()==im,'image identity drift')


def stage_tree(root,files):
    root.mkdir(exist_ok=False)
    for name,body in sorted(files.items()):
        p=relative(name);need(type(body) is bytes,'engine byte contract')
        f=root.joinpath(*p.parts);f.parent.mkdir(parents=True,exist_ok=True)
        with f.open('xb') as w:w.write(body)


def inventory(root):
    paths=sorted(root.rglob('*'));need(not any(p.is_symlink() for p in paths),'unexpected symlink')
    return {p.relative_to(root).as_posix():digest(p.read_bytes()) for p in paths if p.is_file()}


def sandbox(images, source, snapshot, runtime, profile, base_system, prompt, gateway, *, seconds, tools=None, rpc=True, paged_read=False):
    """Run a real Pi process; return scoped external evidence (never an Engine score)."""
    images.verify();need(type(seconds) is int and seconds>0,'session requires bounded wall time')
    need(images.pi=='sha256:'+runtime['pi_image'],'runtime manifest image mismatch')
    need(type(paged_read) is bool and (not paged_read or tools is None or 'read' in tools),'paging cannot add read permission')
    version=docker('run','--rm','--network','none',images.pi,'node','/opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js','--version').stdout.decode().strip()
    need(version==runtime['pi_version'],'Pi version mismatch')
    node=docker('run','--rm','--network','none',images.pi,'node','--version').stdout.decode().strip().removeprefix('v')
    need(node==runtime['node_version'],'Node version mismatch')
    plock=docker('run','--rm','--network','none',images.pi,'cat','/opt/pi/package-lock.json').stdout
    need(digest(plock)==runtime['package_lock_sha'],'runtime package lock changed')
    codeid=digest(canonical({p.name:digest(p.read_bytes()) for p in RUNNER_FILES}))
    need(codeid==runtime['runner_sha'] and digest(canonical(dict(profile=profile,system=base_system,tools=tools,rpc=rpc,paged_read=paged_read)))==runtime['base_config_sha'],'runner or common configuration drift')
    snapshot.recheck(runtime)
    load=snapshot.manifest['load']; hb=dict(snapshot.items); parts=[hb[x].decode() for x in load['prompts']+load['memory']]
    system='\n\n'.join([base_system,*parts])
    # The provider must verify actual outbound prompt fragments, not trust the guest's loaded event.
    need(tuple(parts)==gateway.provider._expected_prompt_fragments,'Harness prompt verification not paired to install')
    spec=dict(system=system,prompt=prompt,profile=profile,tools=tools,rpc=rpc,paged_read=paged_read,token=gateway.token,
              extensions=load['extensions'],skills=load['skills'],templates=[x for x in load['workflows'] if x.endswith('.md')])
    if profile['maxTokens'] > 8192 or hasattr(gateway.provider, 'model_socket_idle_seconds'):
        # The broker buffers verified SSE, so the guest can be idle for the entire model turn.
        spec['provider_timeout_ms'] = (seconds + 15) * 1000
    run=secrets.token_hex(8);name='ee-v3-'+run;side=name+'-relay';vol=name+'-sock'; relay=None;timedout=False
    started=time.monotonic(); report=dict(run=run,image=images.pi,pi=version,node=node,snapshot=snapshot.identity)
    with tempfile.TemporaryDirectory(prefix='ee-v3-input-') as t:
        root=Path(t);stage_tree(root/'engine',source);snapshot.stage(root/'harness',runtime);(root/'spec.json').write_bytes(canonical(spec))
        before=inventory(root);docker('volume','create',vol)
        try:
            relay=Relay(side,images.bridge,vol,gateway);relay.wait()
            extra=['--mount',ro(HERE/'v3_paged_read.mjs','/opt/ee/v3_paged_read.mjs')] if paged_read else []
            options=[*launch(name),'--read-only','--user','12345:12345','--tmpfs','/scratch:rw,nosuid,nodev,mode=1777,size=384m',
                '--tmpfs','/tmp:rw,nosuid,nodev,mode=1777,size=64m','--mount',ro(vol,'/gate','volume'),'--mount',ro(root,'/input'),
                '--mount',ro(HERE/'v3_guest.mjs','/opt/ee/v3_guest.mjs'),'--mount',ro(HERE/'v3_rpc.mjs','/opt/ee/v3_rpc.mjs'),*extra,images.pi,'node','/opt/ee/v3_guest.mjs']
            docker('create',*options,timeout=30)
            view=json.loads(docker('inspect',name).stdout)[0];h=view['HostConfig'];binds={a['Destination']:a['RW'] for a in view['Mounts']}
            expected_mounts=['/input','/gate','/opt/ee/v3_guest.mjs','/opt/ee/v3_rpc.mjs']+(['/opt/ee/v3_paged_read.mjs'] if paged_read else [])
            need(view['Image']==images.pi and view['Config']['User']=='12345:12345' and not h['Privileged'] and not h.get('DeviceRequests') and h['NetworkMode']=='none' and h['ReadonlyRootfs']
                 and h['CapDrop']==['ALL'] and h['SecurityOpt']==['no-new-privileges'] and set(h['Tmpfs'])=={'/scratch','/tmp'} and binds==dict.fromkeys(expected_mounts,False),'unexpected sandbox configuration')
            report['isolation']=dict(network='none',read_only_root=True,user='12345:12345',mount_writeable=binds,gpu=False)
            try: proc=docker('start','-a',name,timeout=seconds,check=False)
            except subprocess.TimeoutExpired: timedout=True;proc=None
        finally:
            docker('rm','-f',name,check=False);docker('rm','-f',side,check=False)
            if relay:
                relay.close()
                report.update(relay_events=relay.count, transport_fault=relay.error,
                              relay_model_socket_idle_seconds=relay.model_socket_idle_seconds)
            report['isolated_state_removed']=(docker('inspect',name,check=False).returncode!=0 and docker('inspect',side,check=False).returncode!=0)
            report['socket_removed']=docker('volume','rm',vol,check=False).returncode==0
        report['unchanged_install']=inventory(root)==before
    report['wall_ms']=round((time.monotonic()-started)*1000);report['provider_rows']=gateway.provider.rows;report['gateway_events']=gateway.events
    report['timed_out']=timedout; report['pi_tool_executions']=[];report['pi_errors']=0
    if proc:
        report['pi_exit']=proc.returncode
        for line in proc.stdout.splitlines():
            if not line.startswith(b'{'):continue
            try:e=json.loads(line)
            except ValueError:continue
            if e.get('type')=='tool_execution_end':report['pi_tool_executions'].append(dict(name=e.get('toolName'),error=e.get('isError')))
            if e.get('type')=='message_end' and isinstance(m:=e.get('message'),dict) and m.get('stopReason')=='error':report['pi_errors']+=1
    report['status']='pi_session_complete' if proc and proc.returncode==0 and not report['pi_errors'] and report['provider_rows'] and report['provider_rows'][-1]['result']=='complete' and report['provider_rows'][-1]['finish']=='stop' and report['unchanged_install'] and report['socket_removed'] and report['isolated_state_removed'] and not report.get('transport_fault') else 'not_completed'
    return report


def runtime_for(images,package_lock,profile,system,tools,*,rpc=True,paged_read=False):
    return dict(pi_version='1.5.2',pi_image=images.pi.removeprefix('sha256:'),node_version='24.13.0',package_lock_sha=digest(package_lock),
      runner_sha=digest(canonical({p.name:digest(p.read_bytes()) for p in RUNNER_FILES})),
      base_config_sha=digest(canonical(dict(profile=profile,system=system,tools=tools,rpc=rpc,paged_read=paged_read))))
