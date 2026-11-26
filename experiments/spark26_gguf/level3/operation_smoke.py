"""Level 3 operation-coverage driver. It does not acquire a Worker or API lease.

`run_cases` can be bound to disposable, already-admitted Run.handler contexts.
Each scenario requires its own segment context; continue and nomination consume
that context. Run this file directly for the offline fixture, never live.
"""
from base64 import b64decode, b64encode
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evolveengine.v3_harness import canonical, digest

OPERATIONS = (
    'develop_stress', 'confirm_performance', 'confirm_cold_build', 'job_cancel',
    'dependency_fetch', 'harness_asset_save', 'harness_asset_restore', 'continue_session',
    'research_reuse', 'finish_exploration', 'artifact_query',
)
ASSET = b'Level 3 operation coverage fixture\n'
IDENTITY = digest(ASSET)


def _need(ok, explanation):
    if not ok:
        raise AssertionError(explanation)


def run_cases(open_case, *, refusal_limit_seconds=5):
    """`open_case(name, asset)` yields (Gateway, valid_args) for an isolated case.

    With a real Worker, provision a qualified candidate and checkpoint, one
    active cancellable confirmation, and a manifested kernel artifact beforehand.
    `valid_args` applies to the wire operation, except the two asset operations
    which use the same content-addressed put/read sequence as the Pi extension.
    """
    rows = []
    for name in OPERATIONS:
        with open_case(name, ASSET) as (gateway, args):
            def call(route, values):
                start = monotonic()
                status, kind, raw = gateway.dispatch('/rpc/'+route, canonical(values), None)
                _need(kind == 'application/json', f'{route}: no JSON receipt')
                return status, json.loads(raw), round((monotonic()-start)*1000)

            route = {'harness_asset_save':'harness_asset_put',
                     'harness_asset_restore':'harness_asset_read'}.get(name,name)
            status, _, elapsed = call(route, {'invalid_fixture_argument': True})
            _need(400 <= status < 500 and elapsed <= refusal_limit_seconds*1000,
                  f'{name}: malformed request not refused promptly ({status}, {elapsed} ms)')
            _need(call('harness_status', {})[0] == 200 or name == 'job_cancel',
                  f'{name}: handler did not remain available after refusal')
            start = monotonic()
            if name == 'harness_asset_save':
                status, result, _ = call(route, dict(identity=IDENTITY,size=len(ASSET)))
                _need(status == 200 and not result['committed'] and result['next_offset'] == 0,
                      'unexpected fresh Harness asset state')
                status, result, _ = call(route, dict(identity=IDENTITY,size=len(ASSET),offset=0,
                                                      data=b64encode(ASSET).decode('ascii')))
                _need(status == 200 and result['committed'] and result['identity'] == IDENTITY,
                      'Harness asset was not committed')
            elif name == 'harness_asset_restore':
                status, result, _ = call(route, dict(identity=IDENTITY,offset=0))
                _need(status == 200 and b64decode(result['data_b64'],validate=True) == ASSET and
                      result['chunk_sha256'] == IDENTITY and result['next_offset'] is None,
                      'Harness asset restore hash differs')
            else:
                status, result, _ = call(route, args)
                _need(status == 200, f'{name}: valid operation did not complete ({status})')
                if name == 'develop_stress':
                    _need(result.get('status') == 'observed' and result.get('scoring') is False,
                          'stress did not produce a non-scoring observation')
                elif name == 'confirm_cold_build':
                    _need(result.get('status') == 'built' and result.get('data', {}).get('build_mode') == 'independent_cold',
                          'cold-build receipt is not bound to a cold build')
                elif name == 'confirm_performance':
                    _need(result.get('status') == 'eligible', 'fixture confirmation did not finish eligible')
                elif name == 'artifact_query':
                    _need(result.get('identity') == args['identity'] and result.get('kernels'),
                          'wrong or empty kernel trace')
                elif name in {'dependency_fetch','research_reuse'}:
                    pin = result.get('identity' if name == 'dependency_fetch' else 'receipt', '')
                    _need(isinstance(pin,str) and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin),
                          f'{name}: missing content-addressed receipt')
                elif name == 'job_cancel':
                    _need(call('job_status', args)[0] == 200, 'cancelled job is no longer observable')
                elif name in {'continue_session','finish_exploration'}:
                    desired = 'continue_after_cleanup' if name == 'continue_session' else 'proposal_then_continue'
                    _need(result['status'] == desired, f'{name}: wrong transition receipt')
            rows.append(dict(tool=name, invalid_status='refused', invalid_ms=elapsed,
                             valid_status=status, valid_ms=round((monotonic()-start)*1000)))
    return rows


if __name__ == '__main__':
    import unittest
    if sys.argv[1:]:
        raise SystemExit('this command runs only the offline suite; takes no arguments')
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_v3_level3_operation_smoke.py')
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
