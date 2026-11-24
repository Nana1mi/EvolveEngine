"""Host-owned content-addressed V3 Harness versions. No performance-based selection.

Single trusted process owns each lineage; caller must not share the storage mount with an
Agent. Commit does not activate. Installing a snapshot never changes the lineage pointer.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from threading import RLock


HEX = re.compile(r'[0-9a-f]{64}\Z')
PART = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}\Z')
RESERVED = {'con','prn','aux','nul', *(f'com{i}' for i in range(1,10)), *(f'lpt{i}' for i in range(1,10))}
RUNTIME_KEYS = {'pi_version', 'pi_image', 'node_version', 'package_lock_sha', 'runner_sha', 'base_config_sha'}
SHA_KEYS = {'pi_image', 'package_lock_sha', 'runner_sha', 'base_config_sha'}
MODES = {'evolving_train', 'fixed_train', 'evaluation'}
MAX_FILES, MAX_BYTES, PER_FILE = 128, 1<<20, 160<<10


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def digest(x): return sha256(x).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def identifier(s):
    require(isinstance(s,str) and HEX.fullmatch(s),'invalid Harness version identity')
    return s


def relative(s):
    require(type(s) is str and 1<=len(s)<=240 and '\\' not in s, 'invalid Harness path')
    parts=s.split('/')
    require(1<=len(parts)<=12 and all(PART.fullmatch(x) and not x.endswith('.') and x.split('.')[0].lower() not in RESERVED for x in parts),'invalid Harness path components')
    return PurePosixPath(*parts)


def runtime_lock(lock):
    require(isinstance(lock,Mapping) and set(lock)==RUNTIME_KEYS, 'incomplete runtime lock')
    out=dict(lock)
    for k,v in out.items():
        if k in SHA_KEYS: identifier(v)
        else:require(isinstance(v,str) and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+',v),'invalid locked version')
    return out


def layout(entries, all_files):
    allowed={'prompts','skills','extensions','workflows','memory'}
    require(isinstance(entries,Mapping) and set(entries)==allowed,'incomplete Harness load layout')
    selected={}
    for category, seq in entries.items():
        require(isinstance(seq,(tuple,list)) and all(type(x) is str for x in seq) and len(set(seq))==len(seq),'layout entries must be ordered and unique')
        out=[]
        for path in seq:
            p=relative(path); require(path in all_files,'uninstalled Harness entry')
            require(p.parts[0]==category and len(p.parts)>=2,'Harness entry outside its category')
            if category=='extensions':require(p.suffix in ('.mjs','.js','.ts'),'invalid extension entry')
            if category in ('prompts','skills','memory'):require(p.suffix in ('.md','.txt'),'expected text entry')
            out.append(path)
        selected[category]=out
    return selected


@dataclass(frozen=True)
class Snapshot:
    identity: str
    manifest_bytes: bytes
    items: tuple

    @property
    def manifest(self):return json.loads(self.manifest_bytes)

    @classmethod
    def make(cls, files, order, runtime):
        require(isinstance(files,Mapping) and len(files)<=MAX_FILES,'invalid file inventory')
        checked={}; case={}; total=0
        for path, contents in files.items():
            p=relative(path)
            require(p.parts[0] in {'prompts','skills','extensions','workflows','memory','deps'},'unsupported top-level Harness path')
            for j in range(1,len(p.parts)+1):
                prefix='/'.join(p.parts[:j]); old=case.setdefault(prefix.casefold(),prefix)
                require(prefix==old,'case-colliding Harness paths')
            require(type(contents) is bytes and len(contents)<=PER_FILE,'invalid Harness source bytes')
            require(b'\0' not in contents,'binary Harness source not permitted')
            contents.decode('utf8',errors='strict')
            total+=len(contents);require(total<=MAX_BYTES,'Harness too large')
            checked[path]=contents
        for path in checked:
            require(not any(parent.as_posix() in checked for parent in relative(path).parents if parent.parts),'file-directory collision')
        load=layout(order,checked)
        rows={p:dict(sha256=digest(b),bytes=len(b)) for p,b in checked.items()}
        payload=canonical(dict(schema=1,files=rows,load=load,runtime=runtime_lock(runtime)))
        return cls(digest(payload),payload,tuple(sorted(checked.items())))

    def recheck(self, runtime):
        m=self.manifest
        original=Snapshot.make(dict(self.items),m['load'],runtime)
        require(original==self,'Harness file inventory, load order or runtime differs from committed version')
        return self

    def stage(self,dst,runtime):
        """Populate an unused host directory; caller binds it read-only into the Agent."""
        self.recheck(runtime); dst=Path(dst);dst.mkdir(exist_ok=False)
        for name,contents in self.items:
            target=dst.joinpath(*relative(name).parts);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(contents)
        self.verify_stage(dst,runtime)
        return dst

    def verify_stage(self,dst,runtime):
        self.recheck(runtime); dst=Path(dst);require(dst.is_dir() and not dst.is_symlink(),'invalid staging root'); checks={}
        for p in dst.rglob('*'):
            meta=p.lstat();require(not p.is_symlink() and (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)),'unsupported staged file')
            if stat.S_ISREG(meta.st_mode):checks[p.relative_to(dst).as_posix()]=p.read_bytes()
        require(checks==dict(self.items),'staged Harness contents differ')
        return self


class Lineage:
    def __init__(self,root,runtime):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.versions=self.root/'versions';self.versions.mkdir(exist_ok=True)
        self.runtime=runtime_lock(runtime);self._lock=RLock()

    def get(self, identity):
        folder=self.versions/identifier(identity)
        require(not folder.is_symlink() and folder.is_dir(),'missing Harness version')
        mfile=folder/'manifest.json'; require(mfile.is_file() and not mfile.is_symlink(),'missing Harness manifest')
        b=mfile.read_bytes();require(digest(b)==identity,'stored Harness manifest hash mismatch');m=json.loads(b)
        tree=folder/'files'; require(tree.is_dir() and not tree.is_symlink(),'missing Harness files')
        items={}
        for p in tree.rglob('*'):
            meta=p.lstat();require(not p.is_symlink() and (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)),'unsafe stored file')
            if stat.S_ISREG(meta.st_mode):
                require(meta.st_size<=PER_FILE,'stored file too large');items[p.relative_to(tree).as_posix()]=p.read_bytes()
        s=Snapshot.make(items,m['load'],self.runtime)
        require(s.manifest_bytes==b,'stored Harness bytes or runtime drift')
        return s

    def _write(self,snapshot):
        snapshot.recheck(self.runtime); dest=self.versions/snapshot.identity
        if dest.exists():self.get(snapshot.identity);return snapshot.identity
        with tempfile.TemporaryDirectory(prefix='stage-',dir=self.root) as d:
            tmp=Path(d)/'object';tmp.mkdir();snapshot.stage(tmp/'files',self.runtime)
            (tmp/'manifest.json').write_bytes(snapshot.manifest_bytes);tmp.rename(dest)
        return snapshot.identity

    @staticmethod
    def assert_train(mode):
        require(mode in MODES,'unknown phase')
        if mode!='evolving_train':raise PermissionError('persistent Harness write disabled in this mode')

    def submit(self,mode,files,layout_spec):
        self.assert_train(mode)
        with self._lock:return self._write(Snapshot.make(files,layout_spec,self.runtime))

    def active(self):
        p=self.root/'active';require(p.is_file() and not p.is_symlink(),'no activated Harness version')
        return self.get(p.read_text(encoding='ascii').strip())

    def _pointer(self,identity):
        with tempfile.TemporaryDirectory(prefix='pointer-',dir=self.root) as d:
            p=Path(d)/'active';p.write_text(identity+'\n',encoding='ascii',newline='\n');os.replace(p,self.root/'active')

    def initialize(self,h0):
        with self._lock:
            require(not (self.root/'active').exists(),'already initialized')
            ref=self._write(h0);self._pointer(ref);return ref

    def activate(self,mode,identity,expected,*,actor,request_id):
        self.assert_train(mode); require(isinstance(actor,str) and actor and isinstance(request_id,str) and request_id,'activation provenance required')
        with self._lock:
            old=self.active().identity
            require(old==identifier(expected),'active version changed before request')
            self.get(identity); self._pointer(identity)
            return dict(old=old,new=identity,actor=actor,request_id=request_id)

    def catalogue(self):
        with self._lock:
            return tuple(sorted(self.get(p.name).identity for p in self.versions.iterdir()))

    def freeze(self):
        """Snapshot by active pointer, never select by a performance/health outcome."""
        with self._lock:return self.active()
