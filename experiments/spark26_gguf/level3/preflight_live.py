"""Authorized live Level 3 Pi/provider smoke on synthetic data; never starts a Worker."""
import argparse
from dataclasses import asdict
from pathlib import Path
import os

from pi_runtime import PI, cfg, profile_and_runtime
from evolveengine.deepseek import Provider
from evolveengine.v3_harness import Snapshot, canonical
from evolveengine.v3_level3_guest import sandbox
from evolveengine.v3_level3_run import EMPTY
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_provider import Limits
from evolveengine.v3_runner import Gateway


SYSTEM = 'This is a synthetic Level 3 transport test. Follow the exact tool instructions and do not modify files.'
PROMPT = ('First call read with path "facts.txt". Then call ee_rpc with operation "facts" and arguments {}. '
          'Call the two tools in this order, in separate turns. Then reply with the two numbers in order.')
LIMITS = Limits(input=16000, output=12288, turns=3, wall_ms=180000, per_turn_output=4096)
COST_RMB = '0.15'
TOOLS = ['read', 'ee_rpc']


def verify(output):
    output.mkdir(exist_ok=False)
    conf, _ = cfg.load_config(cfg.CONFIG)
    fingerprint = cfg.narrow.FINGERPRINT
    provider = Provider('https://api.deepseek.com', 'deepseek-flash', 'DEEPSEEK_API_KEY',
                        'DeepSeek-V4.1-Flash', LIMITS.per_turn_output, fingerprint=fingerprint)
    counter = cfg.narrow.PublicPromptCounter(Path(os.environ['LOCALAPPDATA'])/'EvolveEngine/deepseek-flash-1017-08c8657f-offline')
    budget = cfg.ApiCostBudget(cfg.cost_cap(COST_RMB))
    profile, runtime = profile_and_runtime(LIMITS, SYSTEM, TOOLS)
    source = SourceContract({'facts.txt':b'The local fixture number is 47.\n'})
    engine = source.from_args({'files':{}})
    h0 = Snapshot.make({}, EMPTY, runtime)
    registration = dict(scope='single_bounded_synthetic_api_probe_no_gpu', limits=asdict(LIMITS),
                        expected_fingerprint=fingerprint, estimated_peak_cap_rmb=COST_RMB,
                        runtime=runtime, model=profile, provider_idle_seconds=45)
    cfg.p3.write_once(output/'registration.json', registration)
    result = dict(status='not_passed', scope=registration['scope'], gpu_calls=0)
    lease = None
    called = []
    def rpc(operation, arguments):
        called.append(operation)
        return (200, dict(number=86)) if operation == 'facts' and not arguments else (403, dict(status='denied'))

    try:
        result['provider_listing'] = provider.preflight()
        lease = cfg.ExplorationLease(LIMITS, cfg.provider_transport(provider, 45), counter,
                                     conf['model']['context_window'], output/'model-attempts',
                                     api_cost=budget, model_socket_idle_seconds=60)
        guest = sandbox(PI, source, engine, h0, runtime, profile, SYSTEM, PROMPT, Gateway(lease, rpc),
                        seconds=LIMITS.wall_ms//1000, tools=TOOLS)
        result['session'] = guest
        cfg.p3.need(guest['pi_exit'] == 0 and not guest['pi_errors'] and not guest['transport_fault']
                    and all(guest[k] is True for k in ('isolated_state_removed','socket_removed','unchanged_install')),
                    'guest did not close cleanly')
        cfg.p3.need(guest['pi_tool_executions'] == [dict(name=name,error=False) for name in TOOLS]
                    and called == ['facts'], 'real tools did not complete in order')
        cfg.p3.need([row['prior_tool_contexts'] for row in lease.rows] == [0,1,2]
                    and [row.get('finish') for row in lease.rows] == ['tool_calls','tool_calls','stop']
                    and all(row['usage'] == 'settled' and row['versions'] == [fingerprint] and row['effort'] == 'max'
                            for row in lease.rows) and lease.rows[0]['reasoning_stream_seen'], 'provider evidence mismatch')
        result['status'] = 'pass_scoped'
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        result['failure_type'] = type(exc).__name__
    finally:
        result.update(calls=len(lease.rows) if lease else 0, provider_rows=lease.rows if lease else [],
                      rejections=lease.rejections if lease else [], api_cost=budget.snapshot(), rpc=called)
        cfg.p3.write_once(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized-probe', required=True, type=Path, metavar='NEW_OUTPUT')
    args = parser.parse_args()
    cfg.p3.need(bool(os.environ.get('DEEPSEEK_API_KEY')), 'missing trusted host credential')
    result = verify(args.execute_authorized_probe)
    print(canonical(dict(status=result['status'], calls=result['calls'], api_cost=result['api_cost'],
                         output=str(args.execute_authorized_probe))).decode(), flush=True)
    raise SystemExit(result['status'] != 'pass_scoped')


if __name__ == '__main__':
    main()
