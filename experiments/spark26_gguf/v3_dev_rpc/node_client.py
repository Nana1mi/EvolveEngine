"""Trusted SSH client for the pinned development worker; no Agent credentials or remote shell text."""
from contextlib import contextmanager
from getpass import getpass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os
import re
import socket
from stat import S_ISREG
import sys
import time

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_bytes());ALLOWED=tuple(PLAN['allowed'])
FILES={'PLAN.md':'PLAN.md','node.sh':'node.sh','model.sha256':'model.sha256',
       'service/plan.json':'plan.json','service/service.sh':'service.sh','observer/plan.json':'plan.json',
       'service/residency_hook.py':'residency_hook.py','service/runtime_residency_probe.py':'../runtime_residency_probe.py',
       **{'observer/'+x:x for x in ('probe.py','numeric.py','legacy_eos.py','timing.py')}}

def sha(blob):return sha256(blob).hexdigest()
def enc(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def bundle_file(path):return (HERE/path).read_bytes().replace(b'\r\n',b'\n')
def bundle_manifest():return ''.join(f'{sha(bundle_file(path))}  {name}\n' for name,path in sorted(FILES.items())).encode()
def need(ok,msg):
    if not ok:raise ValueError(msg)


@contextmanager
def connection():
    import paramiko
    pw=getpass('Isolated development node login: ')
    settings=Path.home()/'.ssh/known_hosts';peer='192.168.31.171'; host='192.168.31.108'
    with paramiko.SSHClient() as client, socket.create_connection((host,22),8) as sock:
        client.load_host_keys(str(settings));client.connect(peer,username='jiazheng',password=pw,sock=sock,look_for_keys=False,allow_agent=False,timeout=10,banner_timeout=12,auth_timeout=12)
        try:yield client,pw
        finally:del pw


def invoke(pair,op,edits,engine,source,max_ms):
    need(op in ('develop_quality','develop_measure') and type(max_ms) is int and 60000<=max_ms<=360000,'unsupported or unbounded job');need(re.fullmatch('[a-f0-9]{64}',engine),'engine identity');
    need(set(source)==set(ALLOWED) and set(edits)<=set(source),'unexpected node source paths');base={n:sha(x) for n,x in source.items()};need(set(base.values())=={PLAN['gguf_py']},'base bytes drift')
    patch=enc(dict(operation=op,engine=engine,files={n:x.decode() for n,x in edits.items()},base=base,node_limit_ms=max_ms));input_source=edits.get(ALLOWED[0],source[ALLOWED[0]])
    bundle={name:bundle_file(path) for name,path in FILES.items()};manifest=bundle_manifest();bsha=sha(manifest)
    need(bundle_file('bundle.sha256')==manifest,'trusted worker bundle drift')
    ssh,pw=pair;_,out,err=ssh.exec_command('set -eu; [ "$(hostname)" = spark-c3bc ]; [ "$(id -un)" = jiazheng ]; umask 077; mktemp -d "$HOME/spark26-v3-rpc.XXXXXX"',timeout=20)
    root=out.read().decode().strip();need(out.channel.recv_exit_status()==0 and re.fullmatch(r'/home/jiazheng/spark26-v3-rpc\.[A-Za-z0-9]{6}',root),'not the registered development node')
    local=Path(os.environ.get('EE_V3_RPC_ARCHIVE',str(Path(os.environ['LOCALAPPDATA'])/'EvolveEngine')))/Path(root).name;local.mkdir(exist_ok=False)
    print('node trial',op,local.name,flush=True)
    with ssh.open_sftp() as s:
        for n in ('service','observer'):s.mkdir(root+'/'+n,0o700)
        for n,raw in (bundle|{'bundle.sha256':manifest,'service/patch.json':patch,'service/candidate-gguf.py':input_source}).items():
            t=root+'/'+n;s.putfo(BytesIO(raw),t,len(raw));s.chmod(t,0o600)
    stdin,out,err=ssh.exec_command(f'stty -echo && echo EE_READY && exec bash {root}/node.sh {root} {bsha} {sha(patch)} {sha(input_source)}',get_pty=True,timeout=max_ms/1000+25)
    need(out.readline().strip()=='EE_READY','SSH lease handshake failed');stdin.write(pw+'\n');stdin.flush()
    transport=out.read().decode(errors='replace');rc=out.channel.recv_exit_status();(local/'transport.txt').write_text(transport,encoding='utf8');
    with ssh.open_sftp() as s:
        for d in ('','service','observer','instrumented','phase'):
            if d in ('instrumented','phase') and op!='develop_quality':continue
            if d and d not in s.listdir(root):continue
            if d:(local/d).mkdir(exist_ok=False)
            for e in s.listdir_attr(root+'/'+d):
                n=e.filename
                if not S_ISREG(e.st_mode):continue
                need(re.fullmatch('[a-zA-Z0-9_.-]+',n) and 0<=e.st_size<=36*1024*1024,'unexpected archive member');s.get(root+'/'+d+'/'+n,str(local/d/n))
    print('node finished',op,rc,flush=True)
    return local,rc
