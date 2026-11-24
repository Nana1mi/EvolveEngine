"""Authorized one-session P3 replication using verified outbound source coverage."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic
from unittest.mock import patch

import p3_narrow as narrow
from p3_narrow import p3, node_client, FILE, FINGERPRINT
from evolveengine.v3_provider import GENERATION_TRUNCATED, LeaseError

RUN = 'v3-p3-multiturn-20261124-r01'
PREREG = p3.HERE/'P3-MULTITURN-PREREG-20261124.md'
SOURCE_BYTES = 38532
PAGE_LIMIT = 8192
TOOL_ROUNDS = 3
SYSTEM = narrow.SYSTEM + '''

This run requires the complete original source in your paged read history. Before editing, read the named Engine file sequentially, beginning at offset 0. Use the returned next_offset and expected_version for the next read until truncated is false. Make each page request after seeing the previous result. Then choose your own improvement. Use develop_quality and deliver_engine on the exact candidate in this one session.'''


def parse_object(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def stream_calls(body):
    """Extract tool call identities and observable reasoning boundaries from checked SSE."""
    reasoning, calls = [], {}
    for line in body.splitlines():
        if not line.startswith(b'data:'):
            continue
        event = parse_object(line[5:].strip())
        for choice in event.get('choices', []):
            delta = choice.get('delta') or {}
            thought = delta.get('reasoning_content')
            for tool in delta.get('tool_calls') or []:
                index = tool.get('index')
                if type(index) is not int or index < 0:
                    continue
                item = calls.setdefault(index, dict(prefix=None if thought else len(reasoning), fragments=[]))
                if tool.get('id'):
                    item['id'] = tool['id']
                function = tool.get('function') or {}
                if not isinstance(function, dict):
                    continue
                if type(function.get('name')) is str:
                    item['name'] = function['name']
                if type(function.get('arguments')) is str:
                    item['fragments'].append(function['arguments'])
            if thought:
                reasoning.append(thought)
    return reasoning, {item['id']:dict(name=item.get('name'), arguments=parse_object(''.join(item['fragments'])),
                                      prefix=item['prefix'])
                       for item in calls.values() if 'id' in item}


class Coverage:
    def __init__(self, source):
        p3.need(len(source) == SOURCE_BYTES, 'registered source length mismatch')
        self.source = source

    def page(self, call, result):
        args = call['arguments']
        if call['name'] != 'read' or args.get('path') != FILE or type(result) is not str:
            return None
        try:
            if len(result.encode()) > PAGE_LIMIT:
                return None
            page = parse_object(result)
            offset, content, version = page.get('offset'), page.get('content'), page.get('version')
            if (type(offset) is not int or type(args.get('offset', 0)) is not int or
                    offset != args.get('offset', 0) or not 0 <= offset < len(self.source) or
                    type(content) is not str or type(version) is not str or not re.fullmatch('[a-f0-9]{64}', version) or
                    ('expected_version' in args and args['expected_version'] != version)):
                return None
            blob = content.encode()
            end = offset+len(blob)
            if not blob or end > len(self.source) or self.source[offset:end] != blob:
                return None
            if page.get('truncated') is not (end < len(self.source)) or 'next_offset' not in page:
                return None
            next_offset = page['next_offset']
            if end < len(self.source):
                if type(next_offset) is not int or next_offset != end:
                    return None
            elif next_offset is not None:
                return None
            return offset, end
        except UnicodeError:
            return None

    def inspect(self, messages, completed_turns):
        results = {}
        assistant_groups = set()
        for message in messages:
            if message['role'] == 'assistant' and message.get('tool_calls'):
                assistant_groups.add(frozenset(c['id'] for c in message['tool_calls']))
            if message['role'] == 'tool':
                call_id = message.get('tool_call_id')
                if type(call_id) is str:
                    results.setdefault(call_id, []).append(message.get('content'))
        covered_rounds = 0
        pages = []
        for calls in completed_turns:
            if not calls or frozenset(calls) not in assistant_groups or any(len(results.get(cid, ())) != 1 for cid in calls):
                continue
            covered_rounds += 1
            for cid, call in calls.items():
                page = self.page(call, results[cid][0])
                if page is not None:
                    pages.append(page)
        covered = right = 0
        for start, end in sorted(pages):
            if end > right:
                covered += end-max(right,start)
                right = end
        return dict(tool_rounds=covered_rounds, source_bytes=covered, source_pages=len(pages),
                    met=covered_rounds >= TOOL_ROUNDS and covered == len(self.source))


class ContextLease(narrow.NarrowLease):
    def __init__(self, limits, transport, counter, source):
        self.outbound = transport
        self.coverage = Coverage(source)
        self.calls_by_turn = {}
        self.first_qualified_turn = None
        self.stop_cause = None
        super().__init__(limits, self.observe_transport, counter)

    def observe_transport(self, wire, left):
        row = self.rows[-1]
        prior = [self.calls_by_turn.get(r['turn'], {}) for r in self.rows[:-1]
                 if r.get('finish') == 'tool_calls' and r.get('result') == 'complete']
        row['context'] = self.coverage.inspect(json.loads(wire)['messages'], prior)
        if row['context']['met'] and self.first_qualified_turn is None:
            self.first_qualified_turn = row['turn']
            print(f"Context verified at request {row['turn']} ({row['context']['tool_rounds']} tool rounds, "
                  f"{row['context']['source_bytes']} source bytes)", flush=True)
        return self.outbound(wire, left)

    def _read(self, upstream, ceiling):
        checked = super()._read(upstream, ceiling)
        body, _, _, _, finish, _, _, _ = checked
        if finish != 'tool_calls':
            return checked
        row = self.rows[-1]
        reasoning, calls = stream_calls(body)
        self.calls_by_turn[row['turn']] = calls
        prefixes = []
        for call in calls.values():
            if call['prefix'] is not None and call['name'] and call['arguments']:
                prefix = ''.join(reasoning[:call['prefix']])
                try:
                    tokens = len(self.counter.tokenizer.encode(prefix, add_special_tokens=False).ids)
                except (ValueError, TypeError, RuntimeError):
                    tokens = None
            else:
                tokens = None
            operation = call['arguments'].get('operation') if call['name'] == 'ee_rpc' else None
            prefixes.append(dict(tool=call['name'] if call['name'] in ('read','bash','edit','write','ee_rpc') else 'other',
                                 operation=operation if operation in ('develop_quality','develop_measure','deliver_engine') else None,
                                 public_prefix_tokens=tokens))
        row['tool_call_prefixes'] = prefixes
        return checked

    def exchange(self, raw):
        if self.stop_cause is not None:
            raise LeaseError('session already stopped')
        try:
            reply = super().exchange(raw)
        except LeaseError:
            self.stop_cause = self.rejections[-1]['reason'] if self.rejections else 'provider_failure'
            raise
        if self.rows[-1]['result'] == GENERATION_TRUNCATED:
            self.stop_cause = GENERATION_TRUNCATED
        return reply


def summary(report):
    gate = report.get('first_qualified_turn')
    def after_gate(event):
        return gate is not None and event.get('turn') is not None and event['turn'] >= gate
    accepted = {(r.get('engine'), r.get('receipt_id')) for r in report.get('rpc', [])
                if after_gate(r) and r['operation'] == 'develop_quality' and r['status'] == 'valid'}
    signed = [r for r in report.get('rpc', []) if after_gate(r) and r['operation'] == 'deliver_engine' and
              r['status'] == 'selected_by_agent' and (r.get('engine'),r.get('receipt_id')) in accepted]
    return dict(context_reached=gate is not None, valid_candidates_after_gate=len(accepted),
                accepted_delivery=bool(signed), accepted_delivery_receipt=signed[-1]['receipt_id'] if signed else None)


def run(source, output, counter, transport, worker):
    began = monotonic()
    contract = narrow.NarrowContract(source)
    profile = p3.pilot_profile()
    runtime = p3.runtime_for(p3.PI, (p3.SMOKE/'package-lock.json').read_bytes(), profile, SYSTEM, None, paged_read=True)
    output.mkdir(exist_ok=False)
    base = {FILE:source}
    snapshot = p3.Snapshot.make({}, p3.EMPTY, runtime)
    lineage = p3.Lineage(output/'harness-lineage', runtime)
    initial = lineage.initialize(snapshot)
    prompt = ('First, page through the original gguf.py from byte offset 0 as required. Read each result before '
              'requesting its successor. Once the final page arrives, perform the narrow task. ' + contract.prompt())
    config = dict(run=RUN, prereg_sha=p3.digest(PREREG.read_bytes()), launcher_sha=p3.digest(Path(__file__).read_bytes()),
                  reused_narrow_sha=p3.digest(Path(narrow.__file__).read_bytes()), start_E=p3.source_identity(base), start_H=initial,
                  runtime=runtime, target=narrow.TARGET, source_bytes=SOURCE_BYTES, min_tool_rounds=TOOL_ROUNDS,
                  prompt_sha=p3.digest(prompt.encode()), system_sha=p3.digest(SYSTEM.encode()), profile=profile, fingerprint=FINGERPRINT,
                  limits=asdict(p3.CAP), gpu_node_ms=p3.NODE, development_calls=p3.DEV_CALLS, job_reservation_ms=p3.UNIT)
    p3.write_once(output/'registration.json', config)
    baseline, jobs, rpc = {}, [], []
    report = dict(run=RUN, status='incomplete', stage='same_worker_E0', baseline=baseline, workers=jobs, rpc=rpc, start_H=initial)
    model = policy = cal = None
    try:
        e0 = p3.SourceContract.from_args(contract, {'files':{}})
        cal = worker('develop_measure', {}, e0.digest, base, p3.UNIT, baseline)
        p3.need(cal.status == 'observed', 'same-worker E0 failed')
        p3.metrics(cal.data['scenes'])
        p3.write_once(output/'E0.json', baseline)
        print(f'E0 verified ({cal.node_ms/1000:.3f} node seconds)', flush=True)
        remaining = int(p3.WALL*1000-(monotonic()-began)*1000)
        p3.need(remaining >= p3.UNIT, 'insufficient wall budget after E0')
        model = ContextLease(replace(p3.CAP, wall_ms=remaining), transport, counter, source)

        def backend(operation, artifact, lease):
            job = dict(turn=model.rows[-1]['turn'] if model.rows else None)
            jobs.append(job)
            try:
                result = worker(operation, dict(artifact.edits), artifact.digest, base, lease, job)
                p3.write_once(output/f'development-{len(jobs):02}.json', job)
                if operation == 'develop_measure' and result.status == 'observed':
                    return replace(result, data=dict(result.data, performance_dev=p3.exploratory_index(result.data['scenes'], cal.data['scenes'])))
                return result
            except (ValueError, RuntimeError, OSError) as exc:
                job['error'] = type(exc).__name__
                raise

        policy = p3.Session('evolving_train', lineage, contract, {}, backend, wall_ms=remaining, node_ms=p3.NODE-cal.node_ms,
                            dev_calls=p3.DEV_CALLS, reservations={'develop_quality':p3.UNIT, 'develop_measure':p3.UNIT})

        def handle(operation, args):
            if model.stop_cause:
                return 403, dict(status='session_stopped')
            turn = model.rows[-1]['turn'] if model.rows else None
            previous = len(policy.rows)
            status, reply = policy(operation, args)
            rpc.extend(dict(event, turn=turn) for event in policy.rows[previous:])
            if policy.worker_invalid or status == 502 or reply.get('status') == 'insufficient_evidence':
                model.stop_cause = 'unverified_development_receipt'
            return status, reply

        report['stage'] = 'agent'
        print('One multiturn Pi session started', flush=True)
        report['runner'] = p3.sandbox(p3.PI, base, snapshot, runtime, profile, SYSTEM, prompt, p3.Gateway(model, handle),
                                      seconds=remaining//1000, tools=None, paged_read=True)
        report['status'] = ('session_completed' if report['runner']['status'] == 'pi_session_complete' and
                            not policy.worker_invalid and model.stop_cause is None else 'unsettled')
        report['stage'] = 'finished'
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        report['failure'] = type(exc).__name__
        report['failure_detail'] = str(exc)[:220] if type(exc) is ValueError else None
    finally:
        if model is not None:
            report.update(provider=model.rows, provider_rejections=model.rejections, stop_cause=model.stop_cause,
                          first_qualified_turn=model.first_qualified_turn,
                          usage=dict(input=model.spent_input, output_including_reasoning=model.spent_output))
        if policy is not None:
            report.update(node_ms=policy.spent_node+cal.node_ms, development_calls=p3.DEV_CALLS-policy.calls_left,
                          final_H=lineage.freeze().identity)
            if policy.delivered is not None:
                raw = dict(policy.delivered.edits)[FILE]
                with (output/'delivered-gguf.py').open('xb') as f:
                    f.write(raw)
                report.update(delivered=policy.delivered.digest, delivered_sha256=p3.digest(raw))
        elif cal is not None:
            report['node_ms'] = cal.node_ms
        report['wall_ms'] = round((monotonic()-began)*1000)
        report['acceptance'] = summary(report)
        p3.write_once(output/'receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    opts = parser.parse_args()
    archive = Path(os.environ['LOCALAPPDATA'])/'EvolveEngine'
    source = (archive/'v3-pinned-sglang-source/gguf.py').read_bytes()
    counter = narrow.PublicPromptCounter(archive/'deepseek-flash-1017-08c8657f-offline')
    narrow.prereq(source, archive)
    Coverage(source)
    if not opts.execute:
        print('Local preflight verified', flush=True)
        return
    output = archive/RUN
    p3.need(not output.exists(), 'one-session directory already exists')
    key = sys.stdin.readline(514).rstrip('\r\n')
    password = sys.stdin.readline(514).rstrip('\r\n')
    p3.need(bool(re.fullmatch(r'[A-Za-z0-9_-]{10,512}', key)) and bool(password), 'missing credentials')
    os.environ['DEEPSEEK_API_KEY'] = key
    provider = p3.Provider('https://api.deepseek.com', 'deepseek-flash', 'DEEPSEEK_API_KEY',
                          'DeepSeek-V4.1-Flash', p3.CAP.per_turn_output, fingerprint=FINGERPRINT)
    with patch.object(node_client, 'getpass', lambda _: password), node_client.connection() as ssh:
        result = run(source, output, counter,
                     lambda wire,left:provider.request('/chat/completions',payload=wire,timeout=min(left,180)),
                     narrow.make_worker(ssh, archive))
    print(p3.canonical({k:result[k] for k in ('status','stop_cause','stage','first_qualified_turn','acceptance',
                                             'node_ms','wall_ms','failure') if k in result}).decode(), flush=True)


if __name__ == '__main__':
    main()
