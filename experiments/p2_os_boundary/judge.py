"""Independent synthetic Judge. The host mounts only known-clean Engine bytes."""
import hashlib,json
from pathlib import Path

if __name__=='__main__':
    known={'engine.txt'}
    root=Path('/candidate'); assert {p.name for p in root.iterdir()}==known
    assert not any(Path(p).exists() for p in ['/scratch','/gate','/input','/var/run/docker.sock','/opt/fixture/agent.mjs'])
    data=(root/'engine.txt').read_bytes()
    print(json.dumps(dict(status='pass' if data.decode()=='toy engine 0, no candidate modifications\n' else 'reject',engine_sha=hashlib.sha256(data).hexdigest(),clean_scope=sorted(known),units=2)))
