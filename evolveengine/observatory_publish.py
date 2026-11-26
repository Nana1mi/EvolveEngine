"""Publish only Observatory projections over authenticated SSH; never copy a run root."""
import argparse
from getpass import getpass
from pathlib import Path, PurePosixPath
from time import sleep
from uuid import uuid4

from .observatory import Store
from .observatory_data import RUN_ID


def replace_snapshot(sftp, directory, run, body):
    if not RUN_ID.fullmatch(run) or len(body) > 32 << 20:
        raise ValueError('invalid publication ID or snapshot size')
    final = str(directory / (run + '.json'))
    temporary = str(directory / ('.' + run + '.' + uuid4().hex + '.tmp'))
    try:
        with sftp.open(temporary, 'wb') as f:
            f.write(body)
        sftp.chmod(temporary, 0o600)
        # Put the complete revision into view in one server-side rename.
        sftp.posix_rename(temporary, final)
    finally:
        try:
            sftp.remove(temporary)
        except FileNotFoundError:
            pass


def run_publisher(store, sftp, directory, interval=None):
    published = {}
    while True:
        for row in store.runs():
            name = row['id']
            revision, body = store.snapshot(name)
            if published.get(name) != revision:
                replace_snapshot(sftp, directory, name, body)
                published[name] = revision
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
            run_publisher(Store(live, archive=a.archive), sftp, a.remote_directory, a.watch_seconds)


if __name__ == '__main__':
    main()
