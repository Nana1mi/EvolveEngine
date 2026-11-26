"""Bounded and checksummed multi-frame submission over the existing RPC transport."""
from base64 import b64decode
from binascii import Error
from hashlib import sha256
import json
import re
import secrets
from tempfile import SpooledTemporaryFile

FRAME_BYTES = 192 * 1024
MAX_BYTES = 64 * 1024 * 1024
OPERATIONS = {'candidate_submit': MAX_BYTES, 'harness_submit': 2 * 1024 * 1024,
              'diagnostic_submit': 128 * 1024}


def _check(ok, error):
    if not ok:
        raise ValueError(error)


def _object(pairs):
    out = {}
    for key, value in pairs:
        _check(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def _constant(_):
    raise ValueError('non-finite JSON literal')


class Upload:
    """One serial in-flight upload per untrusted Pi segment; never writes a host path."""
    def __init__(self, dispatch, *, objective=False):
        self.dispatch = dispatch
        self.pending = None
        self.limits = OPERATIONS | ({'diagnostic_submit': 13 << 20} if objective else {})

    def close(self):
        if self.pending is not None:
            self.pending[-1].close()
            self.pending = None

    def __call__(self, op, args):
        if op == 'upload_begin':
            _check(set(args) == {'size', 'operation', 'sha256'}, 'invalid upload begin')
            size, operation, digest = args['size'], args['operation'], args['sha256']
            _check(type(operation) is str and operation in self.limits and type(size) is int
                   and 0 < size <= self.limits[operation] and isinstance(digest, str)
                   and re.fullmatch(r'[0-9a-f]{64}', digest), 'upload admission failed')
            if self.pending is not None:
                key, previous_op, previous_size, previous_digest, stream = self.pending
                _check((operation, size, digest) == (previous_op, previous_size, previous_digest),
                       'another upload is incomplete')
                return 200, {'upload': key, 'next_offset': stream.tell()}
            key = secrets.token_hex(16)
            self.pending = key, operation, size, digest, SpooledTemporaryFile(max_size=2 << 20)
            return 200, {'upload': key, 'next_offset': 0}
        if op not in {'upload_chunk', 'upload_finish', 'upload_cancel'}:
            return self.dispatch(op, args)
        _check(self.pending is not None and args.get('upload') == self.pending[0], 'unknown upload')
        _, operation, expected, digest, body = self.pending
        if op == 'upload_chunk':
            _check(set(args) == {'upload', 'offset', 'data'} and type(args['offset']) is int
                   and args['offset'] == body.tell(), 'out of order upload')
            data = args['data']
            _check(type(data) is str and 0 < len(data) <= 4*((FRAME_BYTES+2)//3), 'invalid frame size')
            try:
                block = b64decode(data, validate=True)
            except (Error, ValueError) as e:
                raise ValueError('invalid frame encoding') from e
            _check(0 < len(block) <= FRAME_BYTES and body.tell()+len(block) <= expected, 'frame range')
            body.write(block)
            return 200, {'next_offset': body.tell()}
        _check(set(args) == {'upload'}, 'invalid upload completion')
        self.pending = None
        with body:
            if op == 'upload_cancel':
                return 200, {'cancelled': True}
            _check(body.tell() == expected, 'upload digest or size mismatch')
            body.seek(0)
            computed = sha256()
            while block := body.read(1 << 20):
                computed.update(block)
            _check(computed.hexdigest() == digest, 'upload digest or size mismatch')
            body.seek(0)
            try:
                payload = json.loads(body.read().decode('utf-8'), object_pairs_hook=_object, parse_constant=_constant)
            except RecursionError as e:
                raise ValueError('upload JSON exceeds depth limit') from e
        _check(type(payload) is dict, 'expected JSON object')
        return self.dispatch(operation, payload)
