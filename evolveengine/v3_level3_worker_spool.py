"""Durable local request/result transport for the authenticated engineering Worker."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from signal import SIGTERM
from subprocess import DEVNULL, Popen
from tempfile import NamedTemporaryFile
from time import monotonic, sleep

from .v3_harness import canonical
from .v3_level3_worker_channel import HEADER, REPLY, REQUEST, _get, _send, need


def _publish(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            need(path.read_bytes() == data, 'durable Worker Job ID already has different content')
    finally:
        temporary.unlink(missing_ok=True)


def _start_ticks(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    return int(raw.rsplit(')', 1)[1].split()[19])


def spool_path(report):
    return report.with_name(report.name + '.transport')


def stop(report, registration):
    root = spool_path(report)
    pid_file = root/'daemon.json'
    if pid_file.is_file() and not (report/'worker-exit.json').is_file():
        row = json.loads(pid_file.read_bytes())
        need(row['registration'] == registration, 'durable Worker daemon has another authorization')
        pid = row['pid']
        try:
            if _start_ticks(pid) == row['start_ticks']:
                os.kill(pid, SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass
        end = monotonic()+30
        while monotonic() < end and not (report/'worker-exit.json').is_file():
            sleep(.1)
        need((report/'worker-exit.json').is_file(), 'durable Worker stop could not be observed')


def _wait(path, report, *, seconds=None):
    deadline = None if seconds is None else monotonic()+seconds
    while not path.is_file():
        need(not (report/'worker-exit.json').is_file(), 'durable Worker exited without this receipt')
        need(deadline is None or monotonic() < deadline, 'durable Worker startup did not publish a greeting')
        sleep(.1)
    return path.read_bytes()


def bridge(report, registration, command, source, sink):
    """Reconnect to a private spool. A bridge exiting leaves accepted jobs running."""
    root = spool_path(report)
    if not root.exists():
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        else:
            _publish(root/'authorization', registration.encode('ascii'))
            with (root/'daemon.log').open('ab', buffering=0) as log:
                Popen(command, stdin=DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    # A reconnect may race the original bridge between mkdir and its authorization write.
    need(root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.getuid(), 'unsafe durable Worker spool')
    approved = _wait(root/'authorization', report, seconds=30)
    need(approved == registration.encode('ascii'), 'durable Worker spool has another authorization')
    greeting = _wait(root/'reply-000000000000', report, seconds=60)
    _send(sink, _get(BytesIO(greeting), REPLY), REPLY)
    while True:
        frame = _get(source, REQUEST)
        number = frame.get('seq')
        need(type(number) is int and 0 < number < 10**12 and frame.get('registration') == registration,
             'invalid durable Worker Job ID')
        key = f'{number:012d}'
        need(number == 1 or (root/f'reply-{number-1:012d}').is_file(), 'nonsequential durable Worker request')
        wire = canonical(frame)
        _publish(root/f'request-{key}', HEADER.pack(len(wire))+wire)
        actual = _wait(root/f'reply-{key}', report)
        response = _get(BytesIO(actual), REPLY)
        _send(sink, response, REPLY)
        if frame['kind'] == 'close' or response['kind'] == 'stopped':
            return


class SpoolIO:
    """Binary stream interface for one long-lived serve() call."""
    def __init__(self, report, registration):
        self.report, self.root = report, spool_path(report)
        self.request_number, self.reply_number = 1, 0
        self.incoming, self.outgoing = BytesIO(), bytearray()
        pid = os.getpid()
        _publish(self.root/'daemon.json', canonical(dict(pid=pid,start_ticks=_start_ticks(pid),registration=registration)))

    def read(self, size):
        block = self.incoming.read(size)
        if block:
            return block
        key = f'{self.request_number:012d}'
        data = _wait(self.root/f'request-{key}', self.report)
        self.incoming = BytesIO(data)
        self.request_number += 1
        return self.incoming.read(size)

    def write(self, data):
        self.outgoing.extend(data)
        return len(data)

    def flush(self):
        key = f'{self.reply_number:012d}'
        _publish(self.root/f'reply-{key}', bytes(self.outgoing))
        if self.reply_number:
            request = (self.root/f'request-{key}').read_bytes()
            _publish(self.root/f'job-{key}.json', canonical(dict(
                job_id=f"{sha256(request).hexdigest()}:{self.reply_number}",
                request_sha256=sha256(request).hexdigest(),
                receipt_sha256=sha256(self.outgoing).hexdigest())))
        self.outgoing.clear()
        self.reply_number += 1
