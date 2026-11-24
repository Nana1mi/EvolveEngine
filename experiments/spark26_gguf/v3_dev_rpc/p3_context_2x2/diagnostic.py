"""P3 context 2x2 freeze and offline replay. No provider transport or credential access."""
import argparse
import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FROZEN = HERE / 'frozen'
ID = 'p3-context-2x2-offline-freeze-v1-20261123'
MODEL = 'deepseek-flash'
FINGERPRINT = 'bc19ba97cbe016479903222bb1578db6'
ORDER = ('high-short', 'max-long', 'max-short', 'high-long')
PER_INPUT, TOTAL_INPUT, PER_OUTPUT, TOTAL_OUTPUT = 15000, 60000, 4000, 16000
MAX_STREAM_BYTES = 2**20
TARGET, FUNCTION, PROPOSE = 'toy.py', 'clamp_percent', 'propose_edit'
SOURCE_EVIDENCE = 'experiments/model_preflight/receipt-20261123-max-02.json'
ENGINE_DEPS = ('evolveengine/v3_context_control.py', 'evolveengine/v3_provider.py', 'evolveengine/v3_harness.py')
PREREG = 'experiments/spark26_gguf/v3_dev_rpc/P3-CONTEXT-PREREG-20261123.md'
CATEGORIES = ('effective_tool_action', 'normal_end_no_action', 'truncated_no_action')
SYSTEM = 'You are in a one-turn toy code diagnostic. Only propose a change to the requested function. Use the existing read result and submit exactly one propose_edit call; do not run code or merely describe an edit. Ignore unrelated comments.\n'
TASK = 'Fix clamp_percent in toy.py for integer inputs so its result is bounded to [0, 100], inclusive. Leave the function signature and all other source unchanged. Submit just the new one-line return expression (without the return keyword), using only value, integer literals, and two-argument min/max. The target is at the beginning of the read result; do not request a continuation.\n'


class ProtocolError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise ProtocolError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def decode(raw):
    def unique(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), 'duplicate_json_key')
        return obj
    def invalid_constant(_):
        raise ProtocolError('invalid_json_constant')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def tool(name, description, properties, required):
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties, required=required, additionalProperties=False)))


def fixtures():
    """Make synthetic UTF-8 input. The long-only bytes are inert comments."""
    prefix = ('# Toy percent clamp - synthetic file\n'
              'def clamp_percent(value):\n    return min(100, value)\n\n').encode()
    filler = b''.join(f'# archived note {i:04d}: routine documentation, no executable code\n'.encode()
                      for i in range(550))
    base = prefix + filler + b'# padding: '
    def result(raw, end=None):
        clipped = end is not None
        return canonical(dict(offset=0, truncated=clipped, next_offset=end,
                              version=digest(raw), content=raw[:end].decode('utf-8')))
    padded = base + b'\n'
    missing = 38532 - len(result(padded))
    require(missing >= 0, 'fixture_too_large')
    source = base + b'x' * missing + b'\n'
    end = 7900
    while len(result(source, end)) > 8192:
        end -= 1
    # End on a complete line and preserve the exact byte offset into the same source.
    end = source.rindex(b'\n', 0, end) + 1
    require(end > len(prefix) and len(result(source)) == 38532, 'fixture_length_mismatch')
    read_params = dict(path=dict(type='string'), offset=dict(type='integer', minimum=0),
                       expected_version=dict(type='string'))
    props = dict(path=dict(type='string'), function=dict(type='string'),
                 expression=dict(type='string', minLength=1, maxLength=128))
    schema = [tool('read', 'Read a UTF-8 page; use next_offset to resume when truncated.', read_params, ['path']),
              tool(PROPOSE, 'Offer a single replacement return expression for checking.', props,
                   ['path', 'function', 'expression'])]
    files = {'system.txt': SYSTEM.encode(), 'task.txt': TASK.encode(),
             'toy-source.txt': source, 'tools.json': canonical(schema)}
    assistant = dict(role='assistant', content='',
        reasoning_content='Synthetic history: read toy.py to locate the target function before proposing a change.',
        tool_calls=[dict(id='call_toy_read', type='function',
                         function=dict(name='read', arguments=canonical(dict(path=TARGET, offset=0)).decode()))])
    for read_size, limit in (('short', end), ('long', None)):
        body = result(source, limit)
        history = [assistant, dict(role='tool', tool_call_id='call_toy_read', content=body.decode())]
        files[f'read-{read_size}.json'] = body
        files[f'history-{read_size}.json'] = canonical(history)
        for effort in ('high', 'max'):
            request = dict(model=MODEL, max_tokens=PER_OUTPUT, stream=True, stream_options=dict(include_usage=True),
                           reasoning_effort=effort, thinking=dict(type='enabled'), tools=schema, tool_choice='auto',
                           messages=[dict(role='system', content=SYSTEM), dict(role='user', content=TASK), *history])
            files[f'{effort}-{read_size}.request.json'] = canonical(request)
    return files


def counter_at(assets):
    # This is the pinned existing public-template counter; it has no hosted transport.
    sys.path.insert(0, str(REPO))
    from evolveengine.v3_context_control import PublicPromptCounter
    return PublicPromptCounter(assets)


def validate_budget(parts, already_input=0, already_output=0):
    require(type(parts) is dict and set(parts) == {'rendered_tokens', 'margin_tokens', 'local_candidate_bound'},
            'reservation_unavailable')
    count, margin, reserve = (parts[k] for k in ('rendered_tokens', 'margin_tokens', 'local_candidate_bound'))
    require(all(type(v) is int and v > 0 for v in (count, margin, reserve))
            and margin == max(512, (count + 9)//10) and reserve == count + margin, 'reservation_unavailable')
    require(reserve <= PER_INPUT and already_input + reserve <= TOTAL_INPUT
            and already_output + PER_OUTPUT <= TOTAL_OUTPUT, 'reservation_failed')
    return reserve


def observe_identity_source(raw):
    evidence = decode(raw)
    turns = evidence['turns']
    require(len(turns) == 3 and all(t['model_requested'] == MODEL and t['models'] == [MODEL] and t['fingerprints'] == [FINGERPRINT]
                                   for t in turns), 'fingerprint_evidence_mismatch')


def build_manifest(files, counter):
    from evolveengine.v3_context_control import ARTIFACTS, MODEL_REVISION, TOKENIZERS_VERSION
    repo_files = {p: digest((REPO / p).read_bytes()) for p in (*ENGINE_DEPS, SOURCE_EVIDENCE, PREREG)}
    observe_identity_source((REPO / SOURCE_EVIDENCE).read_bytes())
    rows, total = [], 0
    for index, name in enumerate(ORDER, 1):
        effort, size = name.split('-')
        request = name + '.request.json'
        parts = counter.reservation(decode(files[request]))
        total += validate_budget(parts, total, (index-1) * PER_OUTPUT)
        rows.append(dict(order=index, cell=name, effort=effort, read_size=size, request=request,
                         history=f'history-{size}.json', tool_result=f'read-{size}.json',
                         read_bytes=len(files[f'read-{size}.json']), reservation=parts, max_tokens=PER_OUTPUT))
    return dict(version=ID, scope='offline_frozen_not_live_authorized', sent_paid_calls=0,
        model=MODEL, expected_fingerprint=FINGERPRINT, fingerprint_evidence=SOURCE_EVIDENCE,
        common=dict(system='system.txt', task='task.txt', source='toy-source.txt', tools='tools.json'),
        cells=rows, frozen_files={p: dict(sha256=digest(raw), bytes=len(raw)) for p, raw in sorted(files.items())},
        scripts={p: digest((HERE / p).read_bytes()) for p in ('diagnostic.py', 'test_diagnostic.py')},
        repo_files=repo_files, counter=dict(revision=MODEL_REVISION, tokenizers=TOKENIZERS_VERSION, files=dict(ARTIFACTS)),
        plan=dict(max_calls=4, retry=False, per_request_input_plan=PER_INPUT, total_input_plan=TOTAL_INPUT,
                  per_request_max_tokens=PER_OUTPUT, total_output_plan=TOTAL_OUTPUT,
                  frozen_input_reservations=total, hosted_usage_bound_attested=False))


def manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def freeze(assets):
    files = fixtures()
    manifest = build_manifest(files, counter_at(assets))
    FROZEN.mkdir(exist_ok=False)  # Never replace an existing freeze.
    for path, raw in files.items():
        (FROZEN / path).write_bytes(raw)
    raw = manifest_bytes(manifest)
    (FROZEN / 'manifest.json').write_bytes(raw)
    return digest(raw)


@dataclass(frozen=True)
class FrozenBundle:
    manifest: dict
    files: dict
    manifest_sha256: str


def verify(assets, expected_sha256, folder=FROZEN):
    raw = (folder / 'manifest.json').read_bytes()
    require(bool(re.fullmatch(r'[0-9a-f]{64}', expected_sha256)) and digest(raw) == expected_sha256,
            'manifest_sha256_mismatch')
    manifest = decode(raw)
    require(manifest['version'] == ID, 'protocol_version_mismatch')
    expected_files = fixtures()
    require(set(manifest['frozen_files']) == set(expected_files), 'fixture_inventory_mismatch')
    require(set(manifest['scripts']) == {'diagnostic.py', 'test_diagnostic.py'} and
            set(manifest['repo_files']) == {*ENGINE_DEPS, SOURCE_EVIDENCE, PREREG}, 'code_inventory_mismatch')
    for hashes, directory in ((manifest['scripts'], HERE), (manifest['repo_files'], REPO)):
        for name, sha in hashes.items():
            require(digest((directory / name).read_bytes()) == sha, 'code_or_evidence_hash_mismatch')
    files = {name: (folder / name).read_bytes() for name in expected_files}
    for name, wire in files.items():
        require(wire == expected_files[name] and digest(wire) == manifest['frozen_files'][name]['sha256'],
                'frozen_file_hash_mismatch:' + name)
    counter = counter_at(assets)
    require(raw == manifest_bytes(build_manifest(files, counter)), 'manifest_rebuild_mismatch')
    return FrozenBundle(manifest, files, expected_sha256), counter


def metered_usage(usage):
    require(type(usage) is dict, 'missing_usage')
    n, m, total = (usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    require(all(type(v) is int and v >= 0 for v in (n,m,total)) and n > 0 and total == n+m, 'invalid_usage')
    details = usage.get('completion_tokens_details')
    require(details is None or type(details) is dict, 'invalid_reasoning_usage')
    reason = details.get('reasoning_tokens') if details else None
    require(reason is None or (type(reason) is int and 0 <= reason <= m), 'invalid_reasoning_usage')
    return dict(input=n, output=m, reasoning=reason)


class Stream:
    def __init__(self, raw):
        self.raw = raw
        self.finish = None
        self.fingerprint_ok = True
        self.usage = None
        self.reasoning = []
        self.at_action = None
        self.ambiguous_boundary = False
        self.calls = {}
        self.issue = None
        self.usage_issue = None
        try:
            self.parse()
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            self.issue = str(exc) if isinstance(exc, ProtocolError) else 'invalid_sse_payload'
        try:
            self.usage = metered_usage(self.usage)
        except ProtocolError as exc:
            self.usage_issue = str(exc)
            self.usage = None

    def parse(self):
        require(type(self.raw) is bytes and 0 < len(self.raw) <= MAX_STREAM_BYTES, 'invalid_sse_size')
        done = False
        meter_seen = False
        for line in self.raw.splitlines():
            if not line or line.startswith(b':'):
                continue
            require(not done and line.startswith(b'data:'), 'invalid_sse_sequence')
            part = line[5:].strip()
            if part == b'[DONE]':
                done = True
                continue
            if meter_seen:
                self.usage = None
                raise ProtocolError('nonterminal_or_duplicate_usage')
            chunk = decode(part)
            require(type(chunk) is dict, 'invalid_sse_chunk')
            self.fingerprint_ok &= ((chunk.get('model'), chunk.get('system_fingerprint')) == (MODEL, FINGERPRINT))
            if chunk.get('usage') is not None:
                meter_seen = True
                self.usage = chunk['usage']
            choices = chunk['choices']
            require(type(choices) is list and len(choices) <= 1, 'invalid_choice_count')
            if not choices:
                require(self.finish is not None and self.usage is not None, 'invalid_usage_chunk')
                continue
            z = choices[0]
            require(type(z) is dict and type(z.get('index')) is int and z['index'] == 0 and self.finish is None,
                    'invalid_choice_sequence')
            d = z['delta']
            require(type(d) is dict, 'invalid_delta')
            reason = d.get('reasoning_content')
            require(reason is None or type(reason) is str, 'invalid_reasoning_delta')
            calls = d.get('tool_calls')
            require(calls is None or type(calls) is list, 'invalid_tool_delta')
            if calls and self.at_action is None:
                self.at_action = len(self.reasoning)
                self.ambiguous_boundary = bool(reason)
            if reason:
                reason.encode('utf-8')  # Reject invalid surrogate data.
                self.reasoning.append(reason)
            for entry in calls or []:
                self.tool_delta(entry)
            end = z['finish_reason']
            if meter_seen and end is None:
                self.usage = None
                raise ProtocolError('nonterminal_usage')
            require(end is None or (type(end) is str and bool(end)), 'malformed_finish_reason')
            if end is not None:
                self.finish = end
        require(done and self.finish is not None, 'incomplete_sse')

    def tool_delta(self, delta):
        ix = delta['index']
        require(type(ix) is int and ix >= 0, 'invalid_tool_index')
        call = self.calls.setdefault(ix, dict(arguments=[]))
        fields = delta.get('function') or {}
        require(type(fields) is dict, 'invalid_tool_function')
        for key, value in (('id', delta.get('id')), ('type', delta.get('type')), ('name', fields.get('name'))):
            if value is not None:
                require(type(value) is str and (key not in call or value == call[key]), 'conflicting_tool_delta')
                call[key] = value
        args = fields.get('arguments')
        if args is not None:
            require(type(args) is str, 'invalid_tool_arguments_delta')
            call['arguments'].append(args)


def checked_expression(node, value):
    """Tiny pure expression interpreter; never execute response-provided code."""
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10000:
        return node.value
    if isinstance(node, ast.Name) and node.id == 'value':
        return value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('min', 'max')
            and len(node.args) == 2 and not node.keywords):
        values = [checked_expression(arg, value) for arg in node.args]
        return (min if node.func.id == 'min' else max)(values)
    raise ProtocolError('expression_not_in_toy_language')


def valid_action(calls):
    if set(calls) != {0}:
        return False
    c = calls[0]
    if not c.get('id') or c.get('type') != 'function' or c.get('name') != PROPOSE:
        return False
    try:
        args = decode(''.join(c['arguments']))
        if (type(args) is not dict or set(args) != {'path','function','expression'}
                or args['path'] != TARGET or args['function'] != FUNCTION):
            return False
        expr = args['expression']
        if type(expr) is not str or not re.fullmatch(r'[0-9a-zA-Z_(), ]{1,128}', expr):
            return False
        tree = ast.parse(expr.strip(), mode='eval').body
        return all(checked_expression(tree, i) == min(max(i, 0), 100)
                   for i in (-10**6, *range(-200,201), 10**6))
    except (ValueError, SyntaxError, RecursionError):
        return False


def classify(stream, counter):
    if stream.finish == 'length':
        category, detail = 'truncated_no_action', 'generation_truncated'
    elif stream.finish == 'tool_calls' and valid_action(stream.calls):
        category, detail = 'effective_tool_action', 'toy_gate_passed'
    elif stream.finish in ('stop', 'tool_calls'):
        category = 'normal_end_no_action'
        detail = 'invalid_tool_attempt' if stream.calls or stream.finish == 'tool_calls' else 'no_tool_attempt'
    else:
        raise ProtocolError('unexpected_finish')
    if category == 'effective_tool_action':
        measured = None if stream.ambiguous_boundary else ''.join(stream.reasoning[:stream.at_action])
        state = 'boundary_unavailable' if measured is None else 'before_effective_action'
    else:
        measured = ''.join(stream.reasoning) or None
        state = 'right_censored_at_response_end'
    # This counts observed text locally, not a split of hosted billing tokens.
    return dict(category=category, detail=detail, prefix_state=state,
                prefix_observed_utf8_bytes=None if measured is None else len(measured.encode()),
                prefix_public_tokenizer_tokens=None if measured is None else
                        len(counter.tokenizer.encode(measured, add_special_tokens=False).ids))


def decision(rows, stopped):
    usable = len(rows) == len(ORDER) and not stopped and all(r.get('category') in CATEGORIES for r in rows)
    successful = [r['cell'] for r in rows if r.get('category') == 'effective_tool_action']
    if not usable:
        return dict(outcome='incomplete_evidence', eligible_cell=None)
    if len(successful) == 1:
        return dict(outcome='eligible_for_separately_authorized_replication', eligible_cell=successful[0])
    if all(r['category'] == 'truncated_no_action' for r in rows):
        return dict(outcome='stop_parameter_screen_shrink_task', eligible_cell=None)
    return dict(outcome='cannot_distinguish_keep_configuration', eligible_cell=None)


class MockUpstream:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.dispatched = []

    def send(self, wire):
        self.dispatched.append(digest(wire))
        return next(self.replies)


def replay(bundle, counter, upstream):
    """Verified frozen bytes -> synthetic reply -> exactly one settlement per dispatch."""
    require(type(upstream) is MockUpstream and not upstream.dispatched, 'fresh_mock_transport_required')
    manifest = bundle.manifest
    cells = manifest['cells']
    require(tuple(c['cell'] for c in cells) == ORDER, 'send_order_mismatch')
    # Whole set is validated before the first dispatch, not piecemeal after paying.
    wires = tuple(bundle.files[c['request']] for c in cells)
    require(all(digest(w) == manifest['frozen_files'][c['request']]['sha256'] for c, w in zip(cells, wires)),
            'request_sha256_mismatch')
    rows = []
    allocated_input = allocated_output = charged_input = charged_output = 0
    stop = None
    for cell, wire in zip(cells, wires):
        row = dict(cell=cell['cell'], request_sha256=digest(wire), request_bytes=len(wire),
                   tool_result_bytes=cell['read_bytes'], dispatched=False)
        rows.append(row)
        try:
            parts = counter.reservation(decode(wire))
            reserve = validate_budget(parts, allocated_input, allocated_output)
            require(parts == cell['reservation'] and decode(wire)['max_tokens'] == PER_OUTPUT,
                    'reservation_drift')
        except (ValueError, KeyError, TypeError, AssertionError, RuntimeError, OSError, IndexError):
            row.update(status='reservation_failed', charged_input=0, charged_output=0)
            stop = row['status']
            break
        allocated_input += reserve
        allocated_output += PER_OUTPUT
        row.update(dispatched=True, input_reserved=reserve, output_reserved=PER_OUTPUT)
        try:
            raw = upstream.send(wire)
        except (StopIteration, OSError):
            raw = None
        stream = Stream(raw)
        actual = stream.usage
        over = actual is not None and (actual['input'] > reserve or actual['output'] > PER_OUTPUT)
        if actual is None or over:
            debit_in = max(reserve, actual['input']) if actual else reserve
            debit_out = max(PER_OUTPUT, actual['output']) if actual else PER_OUTPUT
            row.update(status='usage_uncertain', usage_issue='over_reservation' if over else stream.usage_issue)
        else:
            debit_in, debit_out = actual['input'], actual['output']
            row['status'] = 'fingerprint_mismatch' if not stream.fingerprint_ok else (
                'protocol_invalid' if stream.issue else 'settled')
        charged_input += debit_in
        charged_output += debit_out
        row.update(reported_usage=actual, charged_input=debit_in, charged_output=debit_out,
                   input_overrun=max(0, actual['input']-reserve) if actual else None,
                   output_overrun=max(0, actual['output']-PER_OUTPUT) if actual else None,
                   fingerprint_matches=stream.fingerprint_ok if type(raw) is bytes and raw else None,
                   finish_reason=stream.finish if stream.finish in ('stop','length','tool_calls', 'content_filter',
                                                  'insufficient_system_resource', 'aborted', None) else 'unexpected',
                   protocol_issue=stream.issue, response_sha256=digest(raw) if type(raw) is bytes else None)
        if row['status'] == 'settled':
            try:
                row.update(classify(stream, counter))
            except (ValueError, TypeError, RuntimeError):
                row['status'] = 'classification_unavailable'
        if row['status'] != 'settled':
            stop = row['status']
            break
    return dict(protocol=ID, manifest_sha256=bundle.manifest_sha256, rows=rows, paid_calls=0, gpu_calls=0,
                mock_calls=len(upstream.dispatched), dispatched_request_sha256=upstream.dispatched,
                allocated_input=allocated_input, allocated_output=allocated_output,
                charged_input=charged_input, charged_output=charged_output, stop=stop, decision=decision(rows, stop))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'verify'])
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--expect-sha256')
    opts = p.parse_args()
    if opts.command == 'freeze':
        print(freeze(opts.assets))
    else:
        if not opts.expect_sha256:
            p.error('verify requires the externally recorded manifest SHA-256')
        bundle, _ = verify(opts.assets, opts.expect_sha256)
        print(bundle.manifest_sha256)
