"""Paged access to explicitly published development artifacts on their original host."""
from base64 import b64decode, b64encode
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import os
import re
from shutil import copyfile, disk_usage
import sqlite3
import stat


PAGE_BYTES = 48 * 1024  # fits the authenticated Worker's 128 KiB JSON envelope
KERNEL_PAGE = 20
WRITE_PAGE_BYTES = 192 * 1024


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _check(ok, reason):
    if not ok:
        raise ValueError(reason)


class DevelopmentArtifacts:
    def __init__(self):
        self.files = {}

    def import_profile(self, root, manifest):
        _check(type(manifest) is dict and set(manifest)=={'schema','files'} and
               manifest['schema']=='level3-fixture-trace-files-v1' and type(manifest['files']) is list and
               len(manifest['files'])==2, 'invalid published trace manifest')
        names = {'trace.sqlite':'nsys_sqlite','trace.nsys-rep':'nsys_rep'}
        for entry in manifest['files']:
            _check(type(entry) is dict and set(entry)=={'kind','name','bytes','sha256'} and
                   type(entry['name']) is str and names.pop(entry['name'],None)==entry['kind'] and
                   entry['kind'] in ('nsys_sqlite','nsys_rep') and type(entry['bytes']) is int and
                   entry['bytes']>0 and type(entry['sha256']) is str and len(entry['sha256'])==64,
                   'unregistered published trace file')
            path = root/entry['name']
            _check(path.stat().st_size==entry['bytes'], 'published trace size differs')
            self.publish(path,kind=entry['kind'],label=entry['name'],expected=entry['sha256'])

    def publish(self, path, *, kind, label, expected=None):
        """Only trusted host callers choose paths. Hash once, then bind each read to this inode."""
        path = Path(path)
        _check(kind in ('nsys_sqlite','nsys_rep','diagnostic_stdout','diagnostic_stderr','build_log','harness_asset')
               and isinstance(label,str) and 0 < len(label) <= 160, 'invalid artifact description')
        before = path.lstat()
        _check(stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path.absolute(),
               'development artifact is not a regular host file')
        with path.open('rb') as stream:
            stamp = _identity(before)
            _check(_identity(os.fstat(stream.fileno())) == stamp, 'artifact changed before publication')
            h = sha256()
            for block in iter(lambda:stream.read(1<<20), b''):
                h.update(block)
            _check(_identity(os.fstat(stream.fileno())) == stamp, 'artifact changed during publication')
        pin = h.hexdigest()
        _check(expected is None or expected == pin, 'published artifact hash differs')
        self.files[pin] = (path, stamp, dict(identity=pin, bytes=before.st_size, kind=kind, label=label))
        return pin

    def query(self, view, identity=None, offset=0, limit=None):
        _check(type(offset) is int and offset>=0, 'invalid artifact offset')
        if view == 'list':
            _check(identity is None, 'list does not take an artifact identity')
            count = self._count(limit, 48)
            items = sorted((entry[2] for entry in self.files.values()), key=lambda v:v['identity'])
            rows = items[offset:offset+count]
            return dict(items=rows, next_offset=offset+len(rows) if offset+len(rows)<len(items) else None)
        _check(type(identity) is str and identity in self.files, 'unpublished artifact identity')
        path, stamp, metadata = self.files[identity]
        _check(_identity(path.lstat()) == stamp and stat.S_ISREG(path.lstat().st_mode),
               'published artifact changed')
        if view == 'read':
            count = self._count(limit, PAGE_BYTES)
            size = metadata['bytes']
            _check(offset<=size, 'artifact offset beyond end')
            with path.open('rb') as stream:
                _check(_identity(os.fstat(stream.fileno())) == stamp, 'published artifact changed')
                stream.seek(offset)
                data = stream.read(min(size-offset,count))
                _check(len(data)==min(size-offset,count) and _identity(os.fstat(stream.fileno())) == stamp,
                       'published artifact changed during read')
            end = offset+len(data)
            return dict(identity=identity, offset=offset, bytes=size, data_b64=b64encode(data).decode('ascii'),
                        chunk_sha256=sha256(data).hexdigest(), next_offset=end if end<size else None)
        _check(view == 'kernels' and metadata['kind']=='nsys_sqlite', 'kernel query needs a published Nsight SQLite trace')
        count = self._count(limit, KERNEL_PAGE)
        # The query is fixed; a caller cannot attach a database or address a host path.
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
            db.execute('PRAGMA trusted_schema=OFF')
            db.execute('PRAGMA query_only=ON')
            tables = {name for name,sql in db.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('StringIds','CUPTI_ACTIVITY_KIND_KERNEL')"
            ) if sql and sql.lstrip().upper().startswith('CREATE TABLE')}
            _check(tables == {'StringIds','CUPTI_ACTIVITY_KIND_KERNEL'}, 'missing Nsight kernel tables')
            cursor = db.execute("""
                SELECT substr(s.value,1,256), count(*), sum(k.end-k.start)
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id=k.demangledName
                WHERE k.end>=k.start GROUP BY s.value ORDER BY 3 DESC, s.value LIMIT ? OFFSET ?
            """, (count+1,offset))
            rows = cursor.fetchall()
        _check(_identity(path.lstat())==stamp, 'trace changed during kernel query')
        return dict(identity=identity, offset=offset,
                    kernels=[dict(name=name, calls=calls, device_ns=ns) for name,calls,ns in rows[:count]],
                    next_offset=offset+count if len(rows)>count else None)

    @staticmethod
    def _count(value, maximum):
        if value is None:
            return maximum
        _check(type(value) is int and 1<=value<=maximum, 'invalid artifact page size')
        return value


class HarnessAssets:
    """Run-local, content-addressed guest assets. Only verified complete files are exposed."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.assets = DevelopmentArtifacts()

    def put(self, identity, size, offset=None, data=None):
        _check(type(identity) is str and re.fullmatch('[0-9a-f]{64}', identity) and
               type(size) is int and 0 < size < (1 << 53), 'invalid asset identity or size')
        final, partial = self.root / identity, self.root / (identity + '.partial')
        if final.is_file():
            _check(final.stat().st_size == size, 'committed asset size differs')
            if identity not in self.assets.files:
                self.assets.publish(final, kind='harness_asset', label=identity, expected=identity)
            return dict(identity=identity, size=size, next_offset=size, committed=True)
        written = partial.stat().st_size if partial.is_file() else 0
        _check(written <= size, 'partial asset exceeds submitted size')
        if offset is None and data is None:
            return dict(identity=identity, size=size, next_offset=written, committed=False)
        _check(type(offset) is int and offset == written and isinstance(data, str) and
               0 < len(data) <= 4 * ((WRITE_PAGE_BYTES + 2) // 3), 'invalid asset page')
        block = b64decode(data, validate=True)
        end = written + len(block)
        _check(0 < len(block) <= WRITE_PAGE_BYTES and end <= size, 'asset page exceeds file')
        space = disk_usage(self.root)
        _check(space.free - len(block) >= max(512 << 20, min(4 << 30, space.total // 50)), 'asset disk reserve reached')
        with partial.open('ab') as stream:
            stream.write(block)
        if end == size:
            try:
                self.assets.publish(partial, kind='harness_asset', label=identity, expected=identity)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            partial.replace(final)
            self.assets.publish(final, kind='harness_asset', label=identity, expected=identity)
        return dict(identity=identity, size=size, next_offset=end, committed=end == size)

    def read(self, identity, offset=0, limit=None):
        _check(type(identity) is str and re.fullmatch('[0-9a-f]{64}', identity), 'invalid asset identity')
        if identity not in self.assets.files:
            self.assets.publish(self.root / identity, kind='harness_asset', label=identity, expected=identity)
        return self.assets.query('read', identity, offset, limit)

    def import_objects(self, source, identities):
        for identity in identities:
            _check(type(identity) is str and re.fullmatch('[0-9a-f]{64}', identity), 'invalid resume asset')
            original = source / identity
            _check(original.resolve(strict=True) == original.absolute() and original.is_file(), 'unsafe resume asset')
            target = self.root / identity
            if not target.exists():
                space = disk_usage(self.root)
                _check(space.free - original.stat().st_size >= max(512 << 20, min(4 << 30, space.total // 50)),
                       'asset disk reserve reached')
                temporary = target.with_suffix('.resume')
                try:
                    copyfile(original, temporary)
                    verifier = DevelopmentArtifacts()
                    verifier.publish(temporary, kind='harness_asset', label=identity, expected=identity)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            self.assets.publish(target, kind='harness_asset', label=identity, expected=identity)
            proof = source / f'{identity}.npm.json'
            if proof.is_file() and not proof.is_symlink():
                dest = self.root / proof.name
                if not dest.exists():
                    copyfile(proof,dest)
