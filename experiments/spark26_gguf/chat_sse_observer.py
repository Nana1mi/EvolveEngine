"""Observe one n=1 OpenAI-compatible Chat SSE reply from the isolated E0.

SSE arrival times are client observations, not tokenizer emission timestamps.
Public receipts omit all prompt and response text and volatile call/response IDs.
"""
from dataclasses import dataclass
import hashlib
import json
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

URL = 'http://127.0.0.1:30091/v1/chat/completions'
TIMEOUT_S = 60
LIMIT_LINE = 1 << 17
LIMIT_STREAM = 4 << 20
FINISH = {'stop', 'length', 'tool_calls'}


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON field in stream')
        out[key] = value
    return out


def _bad_constant(_):
    raise ValueError('Non-finite JSON value')


def loads(raw):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_bad_constant)


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _events(stream, now, end):
    """Honor SSE record boundaries, comments, multiple data fields and CRLF."""
    total, data, kind = 0, [], None
    while True:
        raw = stream.readline(LIMIT_LINE + 1)
        t = now()
        if t > end:
            raise ValueError('Chat SSE wall-time budget exceeded')
        if not raw:
            if data:
                raise ValueError('Truncated Chat SSE event')
            return
        total += len(raw)
        if len(raw) > LIMIT_LINE or total > LIMIT_STREAM:
            raise ValueError('Chat SSE size budget exceeded')
        if not raw.endswith(b'\n'):
            raise ValueError('Truncated Chat SSE line')
        line = raw[:-1].removesuffix(b'\r')
        if not line:
            if data:
                yield b'\n'.join(data), kind, t
            data, kind = [], None
        elif not line.startswith(b':'):
            name, sep, value = line.partition(b':')
            value = value.removeprefix(b' ')
            if name == b'data':
                data.append(value)
            elif name == b'event':
                kind = value
            elif name not in {b'id', b'retry'}:
                raise ValueError('Unknown field in pinned Chat SSE wire format')


def _chunk_text(delta, key):
    value = delta.get(key)
    if value is None:
        return ''
    if not isinstance(value, str):
        raise ValueError(f'Malformed Chat delta {key}')
    return value


def _tokens(usage):
    if not isinstance(usage, dict):
        raise ValueError('Chat final usage is missing or malformed')
    prompt, completion, total = (usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens'))
    if not all(type(n) is int and n > 0 for n in (prompt, completion, total)) or prompt+completion != total:
        raise ValueError('Invalid Chat final token accounting')
    return prompt, completion


@dataclass(frozen=True)
class Observation:
    public: dict
    text: str
    reasoning: str
    tools: list


def observe(stream, *, model, start, now=time.perf_counter, timeout=TIMEOUT_S):
    first = visible = last = ended = None
    identity = finish = final_usage = None
    # Digest logical outputs, not SSE partitioning or volatile server IDs.
    chunks, thoughts, tools = [], [], {}
    frames = 0
    for data, kind, t in _events(stream, now, start+timeout):
        if kind not in (None, b'message'):
            raise ValueError('Unsupported or error Chat SSE event')
        if data == b'[DONE]':
            ended = t
            break
        event = loads(data)
        if not isinstance(event, dict) or 'error' in event:
            raise ValueError('Chat stream returned an error or invalid object')
        event_id = event.get('id')
        if ((identity is not None and event_id != identity) or not isinstance(event_id, str) or not event_id
            or event.get('model') != model or event.get('object') != 'chat.completion.chunk'):
            raise ValueError('Chat stream identity changed or is missing')
        identity = event_id
        if final_usage is not None:
            raise ValueError('Extra Chat frame after final usage')
        choices = event.get('choices')
        if choices == []:
            if finish is None:
                raise ValueError('Chat final usage preceded its terminal choice')
            final_usage = _tokens(event.get('usage'))
            continue
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or type(choices[0].get('index')) is not int or choices[0]['index'] != 0 or finish is not None):
            raise ValueError('Expected one unfinished Chat choice with index zero')
        choice = choices[0]
        delta = choice.get('delta')
        if not isinstance(delta, dict) or delta.get('role') not in (None,'assistant'):
            raise ValueError('Malformed Chat delta')
        if event.get('usage') is not None:
            raise ValueError('Unexpected continuous usage; the workload requested final-only usage')
        text = _chunk_text(delta, 'content')
        thought = _chunk_text(delta, 'reasoning_content')
        if text:
            chunks.append(text)
        if thought:
            thoughts.append(thought)
        raw_calls = delta.get('tool_calls')
        if raw_calls is not None and not isinstance(raw_calls, list):
            raise ValueError('Malformed Chat tool calls')
        tool_payload = False
        for call in raw_calls or ():
            if (not isinstance(call, dict) or type(call.get('index')) is not int
                or not 0 <= call['index'] < 64 or call.get('type') not in (None, 'function')):
                raise ValueError('Invalid Chat tool call index or type')
            item = tools.setdefault(call['index'], dict(name='', args=[], id=None))
            if call.get('id') is not None:
                if not isinstance(call['id'], str) or not call['id'] or item['id'] not in (None, call['id']):
                    raise ValueError('Chat tool call ID changed')
                item['id'] = call['id']
            function = call.get('function')
            if function is not None:
                if not isinstance(function, dict):
                    raise ValueError('Malformed Chat tool function')
                name = _chunk_text(function, 'name')
                args = _chunk_text(function, 'arguments')
                # The pinned SGLang emits the full function name, not partial names.
                if name:
                    if item['name'] not in ('',name):
                        raise ValueError('Chat tool name changed')
                    item['name'] = name
                if args:
                    item['args'].append(args)
                tool_payload |= bool(name or args)
        if text or thought or tool_payload:
            first = t if first is None else first
            last = t
            frames += 1
        if visible is None and (text or tool_payload):
            visible = t
        if choice.get('finish_reason') is not None:
            if choice['finish_reason'] not in FINISH:
                raise ValueError('Unexpected terminal Chat choice')
            finish = choice['finish_reason']
    if ended is None or finish is None or final_usage is None or first is None:
        raise ValueError('Incomplete Chat stream; require payload, terminal choice, final usage and [DONE]')
    normalized_tools = []
    if tools or finish == 'tool_calls':
        if finish != 'tool_calls' or not tools or sorted(tools) != list(range(len(tools))):
            raise ValueError('Incomplete Chat tool-call stream')
        for item in (tools[i] for i in range(len(tools))):
            if item['id'] is None or not item['name']:
                raise ValueError('Missing Chat tool ID or name')
            args = loads(''.join(item['args']))
            if not isinstance(args, dict):
                raise ValueError('Chat tool arguments must form an object')
            normalized_tools.append(dict(name=item['name'], arguments=args))
    prompt_tokens, completion_tokens = final_usage
    text, reasoning = ''.join(chunks), ''.join(thoughts)
    def ms(t):
        return round((t-start)*1000,6) if t is not None else None
    return Observation(dict(first_payload_ms=ms(first), first_answer_ms=ms(visible), last_payload_ms=ms(last),
                            done_ms=ms(ended), payload_frames=frames, finish=finish,
                            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            response_sha256=digest([text,reasoning,normalized_tools])),
                       text, reasoning, normalized_tools)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_):
        return None


def request(payload):
    if (payload.get('n',1) != 1 or payload.get('stream') is not True
        or not isinstance(payload.get('model'),str)
        or payload.get('stream_options') != {'include_usage':True}):
        raise ValueError('Chat measurement requires n=1, SSE, model and final-only usage')
    req = Request(URL, json.dumps(payload,ensure_ascii=False).encode('utf-8'), {'Content-Type':'application/json'})
    start = time.perf_counter()
    with build_opener(ProxyHandler({}), _NoRedirect()).open(req,timeout=TIMEOUT_S) as response:
        if response.status != 200 or response.headers.get_content_type() != 'text/event-stream':
            raise ValueError('Expected a successful loopback Chat SSE response')
        return observe(response,model=payload['model'],start=start)
