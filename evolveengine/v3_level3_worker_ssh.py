"""Trusted control-host transport to a fixed engineering Worker on spark-c3bc."""
import base64
from contextlib import contextmanager
from hashlib import sha256
from hmac import compare_digest
from os import environ
from pathlib import PurePosixPath
import re
from shlex import join
from socket import create_connection
from time import monotonic, sleep

from .v3_level3_worker_channel import ChannelIO, WorkerClient, WorkerChannelError

def _worker_target():
    """Use only trusted controller settings; never accept an unpinned target."""
    host = environ.get('EE_L3_WORKER_SSH_HOST', '')
    user = environ.get('EE_L3_WORKER_SSH_USER', '')
    pin = environ.get('EE_L3_WORKER_SSH_ED25519', '')
    if not (re.fullmatch(r'[a-zA-Z0-9.:_-]{1,253}', host)
            and re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', user)
            and re.fullmatch(r'SHA256:[a-zA-Z0-9+/]{43}', pin)):
        raise WorkerChannelError('configure a trusted Worker host, user and Ed25519 fingerprint')
    return host, user, pin.removeprefix('SHA256:')


def _remote_path(raw):
    if not (type(raw) is str and re.fullmatch(r'/var/tmp/[a-zA-Z0-9_./-]{1,245}',raw)
            and str(PurePosixPath(raw)) == raw and '..' not in PurePosixPath(raw).parts):
        raise ValueError('expected registered absolute Worker path')
    return raw


def _worker_command(*, entry, judge_deps, report, registration, authorization, pin, durable=None):
    if type(pin) is not str or not re.fullmatch(r'[a-f0-9]{64}',pin):
        raise ValueError('expected Worker authorization fingerprint')
    entry,judge_deps,report,registration,authorization = map(
        _remote_path,(entry,judge_deps,report,registration,authorization))
    if not entry.endswith('/experiments/spark26_gguf/level3/worker_stdio.py'):
        raise ValueError('expected fixed stdio Worker entrypoint')
    if durable not in (None, 'bridge', 'stop'):
        raise ValueError('unexpected durable Worker endpoint')
    return join(['/usr/bin/sudo','-k','-S','-p','','--','/usr/bin/env','-i',
                 'PATH=/usr/sbin:/usr/bin:/sbin:/bin','LC_ALL=C','PYTHONPATH='+judge_deps,
                 '/usr/bin/python3','-Bu',entry,report,'--registration',registration,
                 '--authorization',authorization,'--sha256',pin] +
                (['--durable',durable] if durable else []))


@contextmanager
def _authenticated_session(credential, command, target):
    import paramiko
    if type(credential) is not str or not credential or any(c in credential for c in '\x00\r\n'):
        raise ValueError('invalid operator credential')
    host, user, host_key = target
    with create_connection((host,22),10) as sock,paramiko.Transport(sock) as transport:
        transport.start_client(timeout=15)
        key = transport.get_remote_server_key()
        fp = base64.b64encode(sha256(key.asbytes()).digest()).decode('ascii').rstrip('=')
        if key.get_name()!='ssh-ed25519' or not compare_digest(fp,host_key):
            raise WorkerChannelError('pinned Worker host key mismatch')
        transport.auth_password(user,credential,fallback=False)
        transport.set_keepalive(30)
        with transport.open_session(timeout=15) as channel:
            channel.settimeout(30)
            channel.exec_command(command)
            channel.sendall((credential+'\n').encode('utf-8'))
            yield channel


@contextmanager
def durable_ssh_worker(contract, *, credential, **args):
    """Reattach to the same Worker and sequence after a dropped SSH connection."""
    import paramiko
    target = _worker_target()
    bridge_command, stop_command = (_worker_command(durable=action, **args) for action in ('bridge','stop'))
    active = None

    def reopen():
        nonlocal active
        if active is not None:
            active.__exit__(None,None,None)
        active = _authenticated_session(credential, bridge_command, target)
        try:
            channel = active.__enter__()
        except (OSError,paramiko.SSHException):
            active = None
            raise OSError('authenticated Worker transport unavailable') from None
        return ChannelIO(channel), ChannelIO(channel), channel.settimeout

    def interrupt():
        with _authenticated_session(credential, stop_command, target) as channel:
            channel.shutdown_write()
            if not wait_for_exit(channel,35) or channel.recv_exit_status() != 0:
                raise WorkerChannelError('durable Worker stop was not acknowledged')

    try:
        stream, out, timeout = reopen()
        worker = WorkerClient(stream,out,contract,args['pin'],set_timeout=timeout,
                              reconnect=reopen,interrupt_transport=interrupt)
        try:
            yield worker
        except BaseException:
            worker.interrupt()
            raise
        else:
            worker.close()
    finally:
        if active is not None:
            active.__exit__(None,None,None)


@contextmanager
def ssh_worker(contract, *, credential, entry, judge_deps, report, registration, authorization, pin):
    """Open one host-key-authenticated binary connection; do not retry any call.

    Optional Paramiko dependency is only imported by the trusted host. The
    credential must come from the operator, never from CLI args or Agent state.
    SSH disconnects do not certify cleanup; a dispatched job remains reserved.
    This does not grant any Agent a general SSH or command execution capability.
    """
    command = _worker_command(entry=entry,judge_deps=judge_deps,report=report,registration=registration,
                              authorization=authorization,pin=pin)
    with _authenticated_session(credential, command, _worker_target()) as channel:
        try:
            client = WorkerClient.from_channel(channel,contract,pin)
            yield client
            client.close()
        except BaseException:
            # A half-close reaches the server's stdin while preserving the
            # receive side for shutdown. Never issue another Worker RPC.
            try:
                channel.shutdown_write()
                wait_for_exit(channel, 25)
            except Exception:
                pass
            raise
        channel.shutdown_write()
        if not wait_for_exit(channel, 15) or channel.recv_exit_status()!=0:
            raise WorkerChannelError('Worker did not exit after verified close')


def wait_for_exit(channel, seconds):
    deadline = monotonic()+seconds
    while not channel.exit_status_ready() and monotonic()<deadline:
        sleep(.05)
    return channel.exit_status_ready()
