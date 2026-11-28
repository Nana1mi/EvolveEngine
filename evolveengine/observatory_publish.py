"""Publish allowlisted Observatory projections and incremental trace bundles over SSH."""
import argparse
from getpass import getpass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import sleep
from uuid import uuid4

from .observatory import Store, encode
from .observatory_data import RUN_ID, is_eligible_run
from . import observatory_detail as detail


def _ensure_remote_dir(sftp, path: PurePosixPath):
    parts = path.parts
    cur = PurePosixPath(parts[0])
    for part in parts[1:]:
        cur = cur / part
        try:
            sftp.stat(str(cur))
        except FileNotFoundError:
            sftp.mkdir(str(cur), mode=0o700)


def _atomic_put(sftp, target: PurePosixPath, body: bytes):
    temporary = str(target.parent / ('.' + target.name + '.' + uuid4().hex + '.tmp'))
    try:
        with sftp.open(temporary, 'wb') as f:
            f.write(body)
        sftp.chmod(temporary, 0o600)
        sftp.posix_rename(temporary, str(target))
    finally:
        try:
            sftp.remove(temporary)
        except FileNotFoundError:
            pass


def replace_snapshot(sftp, directory, run, body):
    if not RUN_ID.fullmatch(run) or len(body) > 32 << 20:
        raise ValueError('invalid publication ID or snapshot size')
    _atomic_put(sftp, directory / (run + '.json'), body)


def sync_bundle(sftp, directory: PurePosixPath, run: str, root: Path, state: dict):
    """Incrementally push allowlisted redacted trace blobs & detail views to <run>.bundle/."""
    ok, _, _ = is_eligible_run(root)
    if not ok:
        return
    bundle = directory / (run + '.bundle')
    trace_remote = bundle / 'trace'
    blobs_remote = trace_remote / 'blobs'
    cands_remote = bundle / 'candidates'
    pages_remote = bundle / 'research-pages'
    for d in (bundle, trace_remote, blobs_remote, cands_remote, pages_remote):
        _ensure_remote_dir(sftp, d)

    uploaded_blobs = state.setdefault('blobs', set())
    if not state.get('blobs_listed'):
        try:
            uploaded_blobs.update(sftp.listdir(str(blobs_remote)))
        except OSError:
            pass
        state['blobs_listed'] = True

    # 1. Sync content-addressed trace blobs & segment JSONLs incrementally
    local_trace = root / 'trace'
    if local_trace.is_dir():
        _, referenced = detail._read_trace_events(root)
        for sha in sorted(referenced):
            fname = f'{sha}.json'
            if fname in uploaded_blobs:
                continue
            blob_path = local_trace / 'blobs' / fname
            if blob_path.is_file():
                _atomic_put(sftp, blobs_remote / fname, blob_path.read_bytes())
                uploaded_blobs.add(fname)
        meta = local_trace / 'run-meta.json'
        if meta.is_file():
            raw = meta.read_bytes()
            h = sha256(raw).hexdigest()
            if state.get('run-meta') != h:
                _atomic_put(sftp, trace_remote / 'run-meta.json', raw)
                state['run-meta'] = h
        seg_sizes = state.setdefault('segments', {})
        for seg_file in sorted(local_trace.glob('segment-*.jsonl')):
            raw = seg_file.read_bytes()
            if seg_sizes.get(seg_file.name) != len(raw):
                _atomic_put(sftp, trace_remote / seg_file.name, raw)
                seg_sizes[seg_file.name] = len(raw)

    # 2. Sync compact allowlisted detail views only when their digest changes
    hashes = state.setdefault('views', {})
    def put_view(rel: PurePosixPath, doc):
        body = encode(doc)
        h = sha256(body).hexdigest()
        key = str(rel)
        if hashes.get(key) != h:
            _atomic_put(sftp, bundle / rel, body)
            hashes[key] = h

    put_view(PurePosixPath('waterfall.json'), detail.waterfall_view(root, run))
    res_doc = detail.research_view(root)
    put_view(PurePosixPath('research.json'), res_doc)
    for page in res_doc.get('pages') or ():
        pid = page.get('id')
        if pid and f'page:{pid}' not in hashes:
            try:
                put_view(PurePosixPath('research-pages') / f'{pid}.json', detail.read_research_page(root, pid))
                hashes[f'page:{pid}'] = 'done'
            except (LookupError, PermissionError):
                pass
    put_view(PurePosixPath('harness.json'), detail.harness_view(root, run))
    put_view(PurePosixPath('context.json'), detail.context_growth_view(root))

    engines, _, _ = detail._load_engine_objects(root)
    for eng in sorted(engines):
        try:
            put_view(PurePosixPath('candidates') / f'{eng}.json', detail.candidate_detail(root, run, eng))
        except (LookupError, PermissionError):
            pass


def run_publisher(store, sftp, directory, interval=None, *, include_bundles=True):
    published = {}
    bundle_states = {}
    while True:
        for row in store.runs():
            name = row['id']
            revision, body = store.snapshot(name)
            if published.get(name) != revision:
                replace_snapshot(sftp, directory, name, body)
                published[name] = revision
            if include_bundles and name in store.live:
                sync_bundle(sftp, directory, name, store.live[name], bundle_states.setdefault(name, {}))
        if interval is None:
            return
        sleep(interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--user', required=True)
    p.add_argument('--remote-directory', required=True, type=PurePosixPath)
    p.add_argument('--run', action='append', default=[], metavar='ID=PATH')
    p.add_argument('--archive', type=Path)
    p.add_argument('--watch-seconds', type=int)
    p.add_argument('--no-bundles', action='store_true', help='publish lean snapshots only without trace/diff bundles')
    p.add_argument('--password-prompt', action='store_true', help='hold the SSH password in memory only')
    a = p.parse_args()
    if (not a.remote_directory.is_absolute() or '..' in a.remote_directory.parts
            or (a.watch_seconds is not None and a.watch_seconds < 5)):
        p.error('use an absolute remote directory and a watch interval of at least 5 seconds')
    live = {}
    for raw in a.run:
        run, sep, path = raw.partition('=')
        if not sep or not path or not RUN_ID.fullmatch(run) or run in live:
            p.error('expected a unique lowercase ID=PATH')
        live[run] = Path(path).resolve()
    import paramiko
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        ssh.connect(a.host, username=a.user, timeout=10, auth_timeout=20,
                    password=getpass('SSH password: ') if a.password_prompt else None,
                    allow_agent=not a.password_prompt, look_for_keys=not a.password_prompt)
        with ssh.open_sftp() as sftp:
            sftp.get_channel().settimeout(20)
            run_publisher(Store(live, archive=a.archive), sftp, a.remote_directory, a.watch_seconds,
                          include_bundles=not a.no_bundles)


if __name__ == '__main__':
    main()
