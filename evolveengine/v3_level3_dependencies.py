"""Fetch pinned public npm packages for read-only installation into the next Pi segment."""
from base64 import b64decode, b64encode
from hashlib import sha256, sha512
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
import tarfile
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .deepseek import NoRedirect
from .v3_harness import canonical, relative

REGISTRY = 'https://registry.npmjs.org'
NAME = re.compile(r'(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*\Z')
VERSION = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.+-]+)?\Z')
SHA = re.compile(r'[a-f0-9]{64}\Z')
LOCK = 'deps/npm-lock.json'
MAX_PACKAGE = 64 << 20
MAX_EXPANDED = 256 << 20


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _get(url, maximum):
    view = urlsplit(url)
    _need(view.netloc == 'registry.npmjs.org' and view.scheme == 'https' and not view.fragment and not view.query,
          'npm dependency URL left the public registry')
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(Request(url), timeout=45) as reply:
            data = reply.read(maximum+1)
    except OSError as exc:
        raise ValueError('public npm package could not be fetched') from exc
    _need(len(data) <= maximum, 'npm package exceeds per-file capacity')
    return data


def fetch(artifact_store, name, version):
    _need(type(name) is str and NAME.fullmatch(name) and type(version) is str and VERSION.fullmatch(version),
          'expected a public npm package and exact version')
    metadata = json.loads(_get(f'{REGISTRY}/{quote(name, safe="")}/{quote(version, safe="")}', 4<<20))
    _need((metadata['name'],metadata['version']) == (name,version), 'registry package identity differs')
    dist = metadata['dist']
    _need(dist['integrity'].startswith('sha512-'), 'npm release needs a published SHA-512 digest')
    package = _get(dist['tarball'], MAX_PACKAGE)
    _need(sha512(package).digest() == b64decode(dist['integrity'][7:], validate=True), 'npm tarball integrity differs')
    address = sha256(package).hexdigest()
    page = artifact_store.put(address,len(package))
    while not page['committed']:
        offset = page['next_offset']
        page = artifact_store.put(address,len(package),offset,b64encode(package[offset:offset+(192<<10)]).decode('ascii'))
    row = dict(name=name, version=version, identity=address, url=dist['tarball'],
               registry_integrity=dist['integrity'], license=metadata.get('license'),
               dependencies=metadata.get('dependencies', {}))
    proof = artifact_store.root / f'{address}.npm.json'
    wire = canonical(row)+b'\n'
    if proof.exists():
        _need(proof.read_bytes() == wire, 'npm release metadata differs from archived receipt')
    else:
        with proof.open('xb') as out:
            out.write(wire)
    return row


def stage(snapshot, destination, assets):
    lock = dict(snapshot.items).get(LOCK)
    if lock is None:
        return
    rows = json.loads(lock)
    _need(type(rows) is list and len(rows) == len({r['name'] for r in rows}), 'invalid npm lock inventory')
    locked = {r['name']: r['version'] for r in rows}
    for row in rows:
        _need(set(row) == {'name','version','identity'}, 'invalid npm lock entry')
        name, version, identity = row['name'],row['version'],row['identity']
        _need(type(name) is str and NAME.fullmatch(name) and type(version) is str and VERSION.fullmatch(version)
              and type(identity) is str and SHA.fullmatch(identity), 'invalid pinned npm package')
        source = assets / identity
        verified = json.loads((assets / f'{identity}.npm.json').read_bytes())
        _need((verified['name'],verified['version'],verified['identity']) == (name,version,identity),
              'npm lock does not match registry receipt')
        _need(source.is_file() and not source.is_symlink() and source.stat().st_size <= MAX_PACKAGE,
              'locked npm dependency is missing from the run store')
        data = source.read_bytes()
        _need(sha256(data).hexdigest() == identity, 'locked npm artifact changed')
        target = destination.joinpath('node_modules', *PurePosixPath(name).parts)
        target.mkdir(parents=True, exist_ok=False)
        total = 0
        with tarfile.open(fileobj=BytesIO(data), mode='r:gz') as archive:
            for entry in archive:
                if entry.isdir() and entry.name.rstrip('/') == 'package':
                    continue
                _need(entry.name.startswith('package/') and (entry.isfile() or entry.isdir()), 'invalid npm tar entry')
                safe = relative(entry.name.removeprefix('package/').rstrip('/'))
                out = target.joinpath(*safe.parts)
                if entry.isdir():
                    out.mkdir(parents=True,exist_ok=True)
                    continue
                total += entry.size
                _need(total <= MAX_EXPANDED, 'npm expansion exceeds per-package capacity')
                out.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(entry) as handle, out.open('xb') as dest:
                    while block := handle.read(1<<20):
                        dest.write(block)
        manifest = json.loads((target/'package.json').read_bytes())
        _need((manifest['name'],manifest['version']) == (name,version), 'locked npm manifest identity differs')
        _need(set(manifest.get('dependencies', {})) <= locked.keys(), 'transitive npm dependency needs an explicit lock entry')
