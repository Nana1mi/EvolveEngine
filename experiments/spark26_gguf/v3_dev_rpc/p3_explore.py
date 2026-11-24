"""One exploratory Engine session with the registered Flash Max profile."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import os
import re
import sys
from threading import Lock
from unittest.mock import patch

import p3_narrow as narrow
from p3_narrow import p3, node_client
from evolveengine.v3_provider import LeaseError
from evolveengine.v3_runner import LONG_MODEL_OUTPUT, LONG_MODEL_REQUEST

CONFIG = p3.HERE / 'exploration.json'


def load_config(path):
    config = p3.record(path)
    model = config['model']
    budget = config['budget']
    transport = config['transport']
    p3.need((model['id'], model['thinking'], model['reasoning_effort']) ==
            ('deepseek-flash', 'enabled', 'max'), 'exploration requires registered Flash Max')
    cap = model['max_output_tokens']
    context = model['context_window']
    idle = transport['upstream_idle_seconds']
    relay_idle = transport.get('relay_socket_idle_seconds', idle)
    p3.need(type(context) is int and 1 <= context <= 1_000_000, 'invalid Flash context window')
    p3.need(type(cap) is int and cap < context and cap <= LONG_MODEL_OUTPUT,
            'invalid or untested exploration completion limit')
    p3.need(type(budget['gpu_operation_seconds']) is int and p3.UNIT <= budget['gpu_operation_seconds']*1000 <= 360_000,
            'GPU operation reservation must fit the registered worker envelope')
    p3.need(type(idle) is int and 1 <= idle <= 1800, 'invalid provider idle timeout')
    p3.need(type(relay_idle) is int and 1 <= relay_idle <= 1800, 'invalid Relay socket idle timeout')
    cost_cap(budget['api_cost_rmb'])
    operation_ms = budget['gpu_operation_seconds']*1000
    p3.need(type(budget['gpu_node_seconds']) is int and budget['gpu_node_seconds']*1000 >= operation_ms and
            type(budget['development_calls']) is int and budget['development_calls'] > 0 and
            type(budget['wall_seconds']) is int and budget['wall_seconds']*1000 > 2*operation_ms,
            'exploration budgets must permit E0 and development')
    limits = p3.Limits(input=budget['input_tokens'], output=budget['output_tokens'],
                       turns=budget['model_requests'], wall_ms=1000*budget['wall_seconds'],
                       per_turn_output=cap, request_bytes=transport['request_bytes'])
    p3.need(limits.request_bytes <= (LONG_MODEL_REQUEST if cap > 8192 else 512*1024),
            'request exceeds the relay profile')
    p3.need(limits.output >= cap, 'session output cannot fit a full model turn')
    term = config['termination']
    p3.need(config['accounting']['record_all_costs'] is True and type(term['stop_on_agent_delivery']) is bool and
            type(term['max_truncation_recoveries']) is int and term['max_truncation_recoveries'] in (0, 1),
            'expected metering and the tested truncation policy')
    config = dict(config, transport=dict(transport, relay_socket_idle_seconds=relay_idle))
    return config, limits


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


# RMB per million tokens from the official Flash table, checked 2026-11-24.
FLASH_RMB_RATES = {'idle': ('0.02', '1', '4'), 'peak': ('0.04', '2', '8')}


def cost_cap(text):
    p3.need(type(text) is str and re.fullmatch(r'[0-9]{1,9}(?:\.[0-9]{1,2})?', text) is not None,
            'API cost cap must be a positive RMB string with at most two decimals')
    value = Decimal(text)
    p3.need(value > 0, 'API cost cap must be positive')
    return value


def peak_uncached_rmb(prompt, completion):
    _, input_rate, output_rate = map(Decimal, FLASH_RMB_RATES['peak'])
    return (prompt*input_rate+completion*output_rate)/Decimal(1_000_000)


class ApiCostBudget:
    """One peak/no-cache exposure limit shared by both Pi segments."""
    def __init__(self, limit):
        self.limit = limit
        self.charged = Decimal(0)
        self.reserved = Decimal(0)
        self.lock = Lock()

    def reserve(self, meta, persist):
        amount = peak_uncached_rmb(meta['input_bound'], meta['output_bound'])
        with self.lock:
            remaining = self.limit-self.charged-self.reserved
            reservation = dict(reserved_rmb=str(amount), remaining_before_rmb=str(max(Decimal(0), remaining)))
            meta['api_cost'] = reservation
            if amount > remaining:
                raise LeaseError('API cost reservation exceeds remaining RMB')
            persist(reservation)
            self.reserved += amount

    def settle(self, row):
        reserved = Decimal(row['request_meta']['api_cost']['reserved_rmb'])
        charge = peak_uncached_rmb(row['input_tokens'], row['output_including_reasoning'])
        with self.lock:
            self.reserved -= reserved
            self.charged += charge
            row['api_cost'] = dict(charged_rmb=str(charge), **self.snapshot())

    def snapshot(self):
        return dict(limit_rmb=str(self.limit), total_charged_rmb=str(self.charged),
                    outstanding_reservation_rmb=str(self.reserved),
                    remaining_rmb=str(max(Decimal(0), self.limit-self.charged-self.reserved)))


def estimate_rmb(usage):
    if not usage or usage.get('usage') != 'settled':
        return None
    cache = usage.get('reported_cache_usage')
    prompt = usage['input_tokens']
    completion = usage['output_including_reasoning']
    quot = Decimal(1_000_000)
    estimates = {}
    for period, rates in FLASH_RMB_RATES.items():
        hit_rate, miss_rate, output_rate = map(Decimal, rates)
        if cache is None:
            low, high = prompt*hit_rate, prompt*miss_rate
        else:
            low = high = cache['hit']*hit_rate + cache['miss']*miss_rate
        estimates[period] = dict(lower=str((low+completion*output_rate)/quot),
                                 upper=str((high+completion*output_rate)/quot))
    return estimates


class ExplorationLease(narrow.NarrowLease):
    # Buffered Chat Completions frames carry metadata on top of each output delta.
    sse_bytes_per_token = 512
    require_full_generation_window = True
    record_generation_timing = True
    finish_reasons = narrow.NarrowLease.finish_reasons | {'insufficient_system_resource', 'aborted'}

    def __init__(self, limits, transport, counter, context_window, directory, *, api_cost,
                 model_socket_idle_seconds, expected_prompt_fragments=()):
        super().__init__(limits, transport, counter, expected_prompt_fragments=expected_prompt_fragments)
        self.context_window = context_window
        self.api_cost = api_cost
        self.model_socket_idle_seconds = model_socket_idle_seconds
        self.directory = directory
        directory.mkdir(exist_ok=False)
        self.attempts = 0

    def _authorize_upstream(self, meta):
        path = self.directory / f'{self.attempts:04}-cost-reservation.json'
        self.api_cost.reserve(meta, lambda reservation: p3.write_once(path, reservation))

    def _settle_upstream(self, row):
        self.api_cost.settle(row)

    def _read(self, upstream, ceiling):
        result = super()._read(upstream, ceiling)
        usage = result[1]
        if type(usage) is dict:
            total = usage.get('prompt_tokens')
            hit = usage.get('prompt_cache_hit_tokens')
            miss = usage.get('prompt_cache_miss_tokens')
            details = usage.get('prompt_tokens_details')
            standard_hit = details.get('cached_tokens') if type(details) is dict else None
            if hit is None and miss is None and type(total) is int and type(standard_hit) is int:
                hit, miss = standard_hit, total - standard_hit
            valid = (type(total) is int and type(hit) is int and type(miss) is int and
                     hit >= 0 and miss >= 0 and hit + miss == total and
                     (standard_hit is None or (type(standard_hit) is int and standard_hit == hit)))
            self.rows[-1]['reported_cache_usage'] = dict(hit=hit, miss=miss) if valid else None
        return result

    def exchange(self, raw):
        self.attempts += 1
        prefix = f'{self.attempts:04}'
        start = utc_now()
        rows, rejections = len(self.rows), len(self.rejections)
        p3.write_once(self.directory / f'{prefix}-start.json', dict(attempt=self.attempts, started_utc=start))
        try:
            return super().exchange(raw)
        finally:
            settled = self.rows[rows] if len(self.rows) > rows else None
            p3.write_once(self.directory / f'{prefix}-result.json', dict(
                attempt=self.attempts, started_utc=start, ended_utc=utc_now(),
                usage=settled, estimated_rmb=estimate_rmb(settled),
                rejection=self.rejections[rejections] if len(self.rejections) > rejections else None,
                cumulative_charge=dict(input_tokens=self.spent_input, output_tokens=self.spent_output)))


def run(source, output, config, limits, counter, transport, worker):
    budget = config['budget']
    api_cost = ApiCostBudget(cost_cap(budget['api_cost_rmb']))
    registration = dict(config, protocol='p3-exploration-384k-v3', official_flash_rmb_per_million=dict(date='2026-11-24',
        order=['cached_input','uncached_input','output'], rates=FLASH_RMB_RATES))

    def lease_factory(active_limits, sender, tokenizer, *, segment, expected_prompt_fragments):
        ledger = 'model-attempts' if segment == 1 else 'model-attempts-recovery'
        return ExplorationLease(active_limits, sender, tokenizer, config['model']['context_window'],
                                output / ledger, api_cost=api_cost,
                                model_socket_idle_seconds=config['transport']['relay_socket_idle_seconds'],
                                expected_prompt_fragments=expected_prompt_fragments)

    report = narrow.run(source, output, counter, transport, worker, run_id=output.name, limits=limits,
                      gpu_node_ms=1000*budget['gpu_node_seconds'], development_calls=budget['development_calls'],
                      launcher=Path(__file__), lease_factory=lease_factory, operation_ms=1000*budget['gpu_operation_seconds'],
                      stop_on_delivery=config['termination']['stop_on_agent_delivery'], registered_config=registration,
                      truncation_recoveries=config['termination']['max_truncation_recoveries'])
    report['api_cost_budget'] = api_cost.snapshot()
    p3.write_once(output/'api-cost-budget.json', report['api_cost_budget'])
    return report


def provider_transport(provider, idle_seconds):
    def send(wire, left):
        return provider.request('/chat/completions', payload=wire, timeout=min(left, idle_seconds))
    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--execute', type=Path, metavar='NEW_OUTPUT_DIR')
    args = parser.parse_args()
    config, limits = load_config(args.config)
    if args.execute is None:
        print(p3.canonical(config).decode())
        return
    output = args.execute.resolve()
    p3.need(output.parent.is_dir() and not output.exists(), 'execution requires a new output directory')
    archive = Path(os.environ['LOCALAPPDATA']) / 'EvolveEngine'
    source = (archive / 'v3-pinned-sglang-source' / 'gguf.py').read_bytes()
    counter = narrow.PublicPromptCounter(archive / 'deepseek-flash-1017-08c8657f-offline')
    narrow.prereq(source, archive)
    key = sys.stdin.readline(514).rstrip('\r\n')
    password = sys.stdin.readline(514).rstrip('\r\n')
    p3.need(bool(re.fullmatch(r'[A-Za-z0-9_-]{10,512}', key)) and bool(password), 'missing credentials')
    os.environ['DEEPSEEK_API_KEY'] = key
    provider = p3.Provider('https://api.deepseek.com', config['model']['id'], 'DEEPSEEK_API_KEY',
                          'DeepSeek-V4.1-Flash', limits.per_turn_output, fingerprint=narrow.FINGERPRINT)
    idle = config['transport']['upstream_idle_seconds']
    with patch.object(node_client, 'getpass', lambda _: password), node_client.connection() as ssh:
        report = run(source, output, config, limits, counter,
                     provider_transport(provider, idle),
                     narrow.make_worker(ssh, archive))
    print(p3.canonical({k:report[k] for k in ('status', 'stop_cause', 'delivered', 'node_ms', 'wall_ms', 'api_cost_budget', 'failure')
                        if k in report}).decode())


if __name__ == '__main__':
    main()
