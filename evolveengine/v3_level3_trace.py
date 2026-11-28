"""Asynchronous, content-addressed host trace recorder for Level 3 exploration.

Strictly host-owned: never mounted into the Pi sandbox, never affects model
settlement or Worker timing, and redacts secret-like patterns before hashing
or writing to disk.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Full, Queue
import re
import secrets
from threading import Event, Lock, Thread
from time import monotonic, time_ns

from .v3_harness import canonical, digest

SCHEMA = 'level3-trace-v1'
MAX_BLOB_BYTES = 2 * 1024 * 1024  # 2 MiB cap per content-addressed blob
PREVIEW_HEAD = 512 * 1024
PREVIEW_TAIL = 16 * 1024
QUEUE_MAX = 4096

_HEX64 = re.compile(r'^[0-9a-f]{64}\Z')
_REDACT_PATTERNS = (
    (re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----'),
     '[REDACTED_PRIVATE_KEY]'),
    (re.compile(r'(?i)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/=-]{8,}'),
     r'\1[REDACTED]'),
    (re.compile(r'\b(Bearer\s+)[A-Za-z0-9._~+/=-]{16,}'),
     r'\1[REDACTED]'),
    (re.compile(r'\bsk-[A-Za-z0-9_-]{16,}\b'),
     '[REDACTED_KEY]'),
    (re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
     '[REDACTED_TOKEN]'),
    (re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
     '[REDACTED_AWS_KEY]'),
    (re.compile(r'''(?i)\b(deepseek_api_key|api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)(\s*[=:]\s*)(['"]?)([^\s'",;}{]{8,})\3'''),
     r'\1\2\3[REDACTED]\3'),
)


def _utc_iso(ns: int | None = None) -> str:
    sec = (time_ns() if ns is None else ns) / 1e9
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat(timespec='milliseconds')


def redact_text(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    out = text
    for secret in extra_secrets:
        if secret and len(secret) >= 8 and secret in out:
            out = out.replace(secret, '[REDACTED]')
    for pattern, repl in _REDACT_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_obj(value, extra_secrets: tuple[str, ...] = (), _depth: int = 0):
    if _depth > 32:
        return '[TRUNCATED_DEPTH]'
    if isinstance(value, str):
        return redact_text(value, extra_secrets)
    if isinstance(value, list):
        return [redact_obj(item, extra_secrets, _depth + 1) for item in value]
    if isinstance(value, tuple):
        return [redact_obj(item, extra_secrets, _depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(k): redact_obj(v, extra_secrets, _depth + 1) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value), extra_secrets)


def _compact_preview(text: str, limit: int = 240) -> str:
    cleaned = ' '.join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit - 1] + '…'


def _extract_pi_result_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get('content')
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get('text'), str):
                    parts.append(item['text'])
            if parts:
                return '\n'.join(parts)
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)
    return '' if result is None else str(result)


def _summarize_tool_args(tool: str, args) -> tuple[str | None, str, dict]:
    """Return (rpc_operation, short_preview, correlation_fields)."""
    rpc_op = None
    corr: dict = {}
    if not isinstance(args, dict):
        return None, _compact_preview(str(args or ''), 160), corr
    if tool == 'bash':
        cmd = args.get('command') or args.get('cmd') or ''
        return None, _compact_preview(str(cmd), 200), corr
    if tool in ('read', 'write', 'edit'):
        path = args.get('path') or args.get('file_path') or ''
        offset = args.get('offset')
        limit = args.get('limit')
        suffix = ''
        if isinstance(offset, int):
            suffix = f':{offset}' + (f'+{limit}' if isinstance(limit, int) else '')
        if isinstance(path, str) and path:
            corr['path'] = path
        return None, _compact_preview(f'{path}{suffix}', 200), corr
    if tool in ('grep', 'find', 'ls'):
        pattern = args.get('pattern') or ''
        path = args.get('path') or ''
        if isinstance(path, str) and path:
            corr['path'] = path
        return None, _compact_preview(f'{pattern} {path}'.strip(), 200), corr
    if tool == 'ee_rpc':
        op = args.get('operation')
        inner = args.get('arguments') if isinstance(args.get('arguments'), dict) else {}
        if isinstance(op, str):
            rpc_op = op
        eng = inner.get('engine')
        if isinstance(eng, str) and _HEX64.fullmatch(eng):
            corr['engine'] = eng
        parent = inner.get('parent')
        if isinstance(parent, str) and _HEX64.fullmatch(parent):
            corr['parent'] = parent
        if op == 'candidate_submit' and isinstance(inner.get('paths'), list):
            paths = [str(p) for p in inner['paths'][:8]]
            corr['paths'] = paths
            return rpc_op, _compact_preview(f"candidate_submit {', '.join(paths)}", 200), corr
        if op == 'research_search':
            return rpc_op, _compact_preview(f"search: {inner.get('query', '')}", 200), corr
        if op == 'research_fetch':
            return rpc_op, _compact_preview(f"fetch: {inner.get('url', '')}", 200), corr
        if op == 'harness_submit' and isinstance(inner.get('paths'), list):
            return rpc_op, _compact_preview(f"harness_submit {', '.join(map(str, inner['paths'][:6]))}", 200), corr
        if op:
            summary = f'{op}' + (f' {eng[:10]}' if 'engine' in corr else '')
            return rpc_op, _compact_preview(summary, 200), corr
    return None, _compact_preview(json.dumps(args, ensure_ascii=False, default=str), 160), corr


def _extract_rpc_result_ids(rpc_op: str | None, result_text: str) -> dict:
    corr: dict = {}
    if not rpc_op or not result_text:
        return corr
    try:
        parsed = json.loads(result_text)
    except (ValueError, RecursionError):
        return corr
    if not isinstance(parsed, dict):
        return corr
    if isinstance(parsed.get('status'), int):
        corr['rpc_status'] = parsed['status']
    body = parsed.get('result') if isinstance(parsed.get('result'), dict) else parsed
    if not isinstance(body, dict):
        return corr
    for key in ('engine', 'parent', 'build', 'evidence', 'checkpoint', 'version', 'receipt'):
        val = body.get(key)
        if isinstance(val, str) and _HEX64.fullmatch(val):
            corr[key] = val
    if isinstance(body.get('status'), str) and len(body['status']) <= 64:
        corr['op_status'] = body['status']
    if isinstance(body.get('job'), str) and len(body['job']) <= 64:
        corr['job'] = body['job']
    return corr


def parse_sse_response(raw_sse: bytes) -> dict:
    """Reconstruct reasoning, assistant text, and tool_calls from verified SSE bytes."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish = None
    model = None
    for line in raw_sse.splitlines():
        if not line.startswith(b'data:'):
            continue
        piece = line[5:].strip()
        if not piece or piece == b'[DONE]':
            continue
        try:
            chunk = json.loads(piece)
        except (ValueError, RecursionError):
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get('model'), str):
            model = chunk['model']
        for choice in chunk.get('choices') or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta') or {}
            if isinstance(delta, dict):
                rc = delta.get('reasoning_content')
                if isinstance(rc, str) and rc:
                    reasoning_parts.append(rc)
                ct = delta.get('content')
                if isinstance(ct, str) and ct:
                    content_parts.append(ct)
                for tc in delta.get('tool_calls') or ():
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get('index', 0)
                    if not isinstance(idx, int) or idx < 0:
                        continue
                    slot = tool_calls.setdefault(idx, {
                        'index': idx, 'id': None, 'type': 'function',
                        'name': '', 'arguments_parts': [],
                    })
                    if isinstance(tc.get('id'), str) and tc['id']:
                        slot['id'] = tc['id']
                    if isinstance(tc.get('type'), str) and tc['type']:
                        slot['type'] = tc['type']
                    fn = tc.get('function')
                    if isinstance(fn, dict):
                        if isinstance(fn.get('name'), str) and fn['name']:
                            slot['name'] += fn['name']
                        if isinstance(fn.get('arguments'), str) and fn['arguments']:
                            slot['arguments_parts'].append(fn['arguments'])
            if isinstance(choice.get('finish_reason'), str):
                finish = choice['finish_reason']
    calls = []
    for idx in sorted(tool_calls):
        slot = tool_calls[idx]
        raw_args = ''.join(slot['arguments_parts'])
        parsed_args = None
        if raw_args:
            try:
                parsed_args = json.loads(raw_args)
            except (ValueError, RecursionError):
                parsed_args = None
        cid = slot['id']
        calls.append({
            'id': cid,
            'call_sha256': digest(cid.encode('utf-8')) if isinstance(cid, str) and cid else None,
            'name': slot['name'],
            'arguments': parsed_args if parsed_args is not None else raw_args,
        })
    return {
        'model': model,
        'finish': finish,
        'reasoning': ''.join(reasoning_parts),
        'content': ''.join(content_parts),
        'tool_calls': calls,
    }


def _bounded_blob_bytes(payload) -> bytes:
    raw = canonical(payload)
    if len(raw) <= MAX_BLOB_BYTES:
        return raw
    text = raw.decode('utf-8', errors='replace')
    wrapped = {
        '_truncated': True,
        'original_bytes': len(raw),
        'preview': text[:PREVIEW_HEAD],
        'tail': text[-PREVIEW_TAIL:],
    }
    return canonical(wrapped)


class TraceRecorder:
    """Non-blocking background writer for redacted, content-addressed Level 3 traces."""

    def __init__(self, root: Path | str, *, started_utc_ns: int | None = None,
                 extra_secrets: tuple[str, ...] = ()):
        self.root = Path(root)
        self.blobs_dir = self.root / 'blobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.started_utc_ns = time_ns() if started_utc_ns is None else int(started_utc_ns)
        self._origin_mono = monotonic()
        self._secrets_lock = Lock()
        initial_secrets = [s for s in extra_secrets if isinstance(s, str) and len(s) >= 8]
        env_key = os.environ.get('DEEPSEEK_API_KEY')
        if isinstance(env_key, str) and len(env_key) >= 8:
            initial_secrets.append(env_key)
        self._extra_secrets: tuple[str, ...] = tuple(dict.fromkeys(initial_secrets))
        self._queue: Queue = Queue(maxsize=QUEUE_MAX)
        self._closed = False
        self._lock = Lock()
        self._known_blobs: set[str] = set()
        self._segment_seqs: dict[int, int] = {}
        self._open_tools: dict[tuple[int, str], tuple[int, str | None]] = {}
        self.dropped = 0
        self.errors = 0
        self._write_run_meta()
        self._thread = Thread(target=self._run, name='level3-trace-writer', daemon=True)
        self._thread.start()

    def _write_run_meta(self):
        meta_path = self.root / 'run-meta.json'
        if not meta_path.exists():
            tmp = meta_path.with_suffix('.tmp')
            tmp.write_bytes(canonical({
                'schema': SCHEMA,
                'started_utc_ns': self.started_utc_ns,
                'created_utc': _utc_iso(self.started_utc_ns),
            }) + b'\n')
            os.replace(tmp, meta_path)

    def register_secret(self, secret: str | None):
        if not isinstance(secret, str) or len(secret) < 8:
            return
        with self._secrets_lock:
            if secret not in self._extra_secrets:
                self._extra_secrets = (*self._extra_secrets, secret)

    def _secrets(self) -> tuple[str, ...]:
        with self._secrets_lock:
            return self._extra_secrets

    def _elapsed_ms(self, utc_ns: int) -> int:
        return max(0, round((utc_ns - self.started_utc_ns) / 1_000_000))

    def _enqueue(self, item) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except Full:
            self.dropped += 1
            return False

    def _put_blob(self, obj) -> tuple[str, int]:
        redacted = redact_obj(obj, self._secrets())
        raw = _bounded_blob_bytes(redacted)
        h = sha256(raw).hexdigest()
        if h not in self._known_blobs:
            dest = self.blobs_dir / f'{h}.json'
            if not dest.exists():
                tmp = self.blobs_dir / f'.{h}.{secrets.token_hex(4)}.tmp'
                try:
                    tmp.write_bytes(raw)
                    os.replace(tmp, dest)
                finally:
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
            self._known_blobs.add(h)
        return h, len(raw)

    def _append_segment_event(self, segment: int, event: dict):
        seq = self._segment_seqs.get(segment, 0) + 1
        self._segment_seqs[segment] = seq
        record = {'seq': seq, 'segment': segment, **event}
        path = self.root / f'segment-{segment:04d}.jsonl'
        line = canonical(record) + b'\n'
        with path.open('ab') as handle:
            handle.write(line)
            handle.flush()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == 'flush':
                    payload.set()
                elif kind == 'model_exchange':
                    self._handle_model_exchange(payload)
                elif kind == 'pi_event':
                    self._handle_pi_event(payload)
                elif kind == 'worker_event':
                    self._handle_worker_event(payload)
            except Exception:
                self.errors += 1
            finally:
                self._queue.task_done()

    def _handle_model_exchange(self, p: dict):
        seg = int(p['segment'])
        raw_req = p.get('raw_request')
        raw_sse = p.get('raw_sse')
        message_refs = []
        if isinstance(raw_req, bytes) and raw_req:
            try:
                req = json.loads(raw_req)
            except (ValueError, RecursionError):
                req = None
            if isinstance(req, dict) and isinstance(req.get('messages'), list):
                for msg in req['messages']:
                    if not isinstance(msg, dict):
                        continue
                    blob_id, blob_bytes = self._put_blob(msg)
                    ref = {'role': str(msg.get('role') or 'unknown'), 'blob': blob_id, 'bytes': blob_bytes}
                    if isinstance(msg.get('tool_call_id'), str):
                        ref['tool_call_id'] = msg['tool_call_id']
                        ref['call_sha256'] = digest(msg['tool_call_id'].encode('utf-8'))
                    if isinstance(msg.get('tool_calls'), list):
                        cids = [tc.get('id') for tc in msg['tool_calls'] if isinstance(tc, dict) and isinstance(tc.get('id'), str)]
                        if cids:
                            ref['tool_call_ids'] = cids
                    message_refs.append(ref)

        response_blob = None
        response_bytes = 0
        tool_call_ids: list[str] = []
        call_sha256s: list[str] = []
        reasoning_chars = 0
        content_chars = 0
        content_preview = ''
        if isinstance(raw_sse, bytes) and raw_sse:
            parsed = parse_sse_response(raw_sse)
            reasoning_chars = len(parsed['reasoning'])
            content_chars = len(parsed['content'])
            content_preview = _compact_preview(redact_text(parsed['content'], self._secrets()), 240)
            for tc in parsed['tool_calls']:
                if tc.get('id'):
                    tool_call_ids.append(tc['id'])
                if tc.get('call_sha256'):
                    call_sha256s.append(tc['call_sha256'])
            response_blob, response_bytes = self._put_blob(parsed)

        usage = p.get('usage') or {}
        cache = usage.get('reported_cache_usage') if isinstance(usage, dict) else None
        gen = usage.get('generation_timing') if isinstance(usage, dict) else None
        utc_ns = int(p['utc_ns'])
        event = {
            'kind': 'model_exchange',
            'attempt': int(p['attempt']),
            'utc_ns': utc_ns,
            'elapsed_ms': self._elapsed_ms(utc_ns),
            'started_utc': p.get('started_utc'),
            'ended_utc': p.get('ended_utc'),
            'wall_ms': usage.get('wall_ms') if isinstance(usage, dict) else None,
            'finish': usage.get('finish') if isinstance(usage, dict) else None,
            'result': usage.get('result') if isinstance(usage, dict) else None,
            'usage_status': usage.get('usage') if isinstance(usage, dict) else None,
            'input_tokens': usage.get('input_tokens') if isinstance(usage, dict) else None,
            'output_tokens': usage.get('output_including_reasoning') if isinstance(usage, dict) else None,
            'reasoning_tokens': usage.get('reasoning_tokens') if isinstance(usage, dict) else None,
            'cache_hit_tokens': cache.get('hit') if isinstance(cache, dict) else None,
            'cache_miss_tokens': cache.get('miss') if isinstance(cache, dict) else None,
            'first_delta_ms': gen.get('first_delta_ms') if isinstance(gen, dict) else None,
            'last_delta_ms': gen.get('last_delta_ms') if isinstance(gen, dict) else None,
            'output_span_ms': gen.get('output_span_ms') if isinstance(gen, dict) else None,
            'upstream_sha': usage.get('upstream_sha') if isinstance(usage, dict) else None,
            'prompt_fragments': usage.get('prompt_fragments') if isinstance(usage, dict) else None,
            'request_message_count': len(message_refs),
            'message_blobs': message_refs,
            'response_blob': response_blob,
            'response_bytes': response_bytes,
            'reasoning_chars': reasoning_chars,
            'content_chars': content_chars,
            'content_preview': content_preview,
            'tool_call_ids': tool_call_ids,
            'call_sha256s': call_sha256s,
            'rejection': p.get('rejection'),
        }
        self._append_segment_event(seg, event)

    def _handle_pi_event(self, p: dict):
        seg = int(p['segment'])
        utc_ns = int(p['utc_ns'])
        row = p['row']
        ev_type = row.get('type')
        if ev_type == 'tool_execution_start':
            name = row.get('toolName')
            if not isinstance(name, str) or len(name) > 128:
                return
            tool_id = row.get('toolCallId') if isinstance(row.get('toolCallId'), str) else None
            call_sha = digest(tool_id.encode('utf-8')) if tool_id else None
            args = row.get('args')
            rpc_op, preview, corr = _summarize_tool_args(name, args)
            args_blob, args_bytes = self._put_blob({'tool': name, 'args': args})
            if call_sha:
                self._open_tools[(seg, call_sha)] = (utc_ns, rpc_op)
            event = {
                'kind': 'pi_tool_start',
                'utc_ns': utc_ns,
                'elapsed_ms': self._elapsed_ms(utc_ns),
                'tool': name,
                'tool_call_id': tool_id,
                'call_sha256': call_sha,
                'rpc_operation': rpc_op,
                'args_preview': redact_text(preview, self._secrets()),
                'args_blob': args_blob,
                'args_bytes': args_bytes,
                **corr,
            }
            self._append_segment_event(seg, event)
        elif ev_type == 'tool_execution_end':
            name = row.get('toolName')
            if not isinstance(name, str) or len(name) > 128:
                return
            tool_id = row.get('toolCallId') if isinstance(row.get('toolCallId'), str) else None
            call_sha = digest(tool_id.encode('utf-8')) if tool_id else None
            started = self._open_tools.pop((seg, call_sha), None) if call_sha else None
            start_ns, rpc_op = started if started is not None else (None, None)
            duration_ms = round((utc_ns - start_ns) / 1_000_000) if start_ns is not None and utc_ns >= start_ns else None
            result = row.get('result')
            text = _extract_pi_result_text(result)
            corr = _extract_rpc_result_ids(rpc_op, text)
            res_blob, res_bytes = self._put_blob({'tool': name, 'rpc_operation': rpc_op, 'is_error': bool(row.get('isError')), 'result': result})
            event = {
                'kind': 'pi_tool_end',
                'utc_ns': utc_ns,
                'elapsed_ms': self._elapsed_ms(utc_ns),
                'duration_ms': duration_ms,
                'tool': name,
                'tool_call_id': tool_id,
                'call_sha256': call_sha,
                'rpc_operation': rpc_op,
                'is_error': bool(row.get('isError')),
                'result_preview': redact_text(_compact_preview(text, 240), self._secrets()),
                'result_blob': res_blob,
                'result_bytes': res_bytes,
                **corr,
            }
            self._append_segment_event(seg, event)
        elif ev_type == 'message_end' and isinstance(row.get('message'), dict):
            msg = row['message']
            role = str(msg.get('role') or 'unknown')
            stop_reason = msg.get('stopReason') if isinstance(msg.get('stopReason'), str) else None
            blob_id, blob_bytes = self._put_blob(msg)
            event = {
                'kind': 'pi_message',
                'utc_ns': utc_ns,
                'elapsed_ms': self._elapsed_ms(utc_ns),
                'role': role,
                'stop_reason': stop_reason,
                'message_blob': blob_id,
                'message_bytes': blob_bytes,
            }
            self._append_segment_event(seg, event)

    def _handle_worker_event(self, p: dict):
        seg = int(p.get('segment', 0))
        utc_ns = int(p['utc_ns'])
        event = {k: v for k, v in p.items() if k != 'segment'}
        event.setdefault('elapsed_ms', self._elapsed_ms(utc_ns))
        self._append_segment_event(seg, event)

    def record_model_exchange(self, *, segment: int, attempt: int, started_utc: str, ended_utc: str,
                              raw_request: bytes | None, raw_sse: bytes | None,
                              usage: dict | None, rejection: dict | None):
        self._enqueue(('model_exchange', {
            'segment': segment,
            'attempt': attempt,
            'utc_ns': time_ns(),
            'started_utc': started_utc,
            'ended_utc': ended_utc,
            'raw_request': raw_request,
            'raw_sse': raw_sse,
            'usage': dict(usage) if isinstance(usage, dict) else None,
            'rejection': dict(rejection) if isinstance(rejection, dict) else None,
        }))

    def record_pi_event(self, *, segment: int, utc_ns: int, row: dict):
        if row.get('type') not in ('tool_execution_start', 'tool_execution_end', 'message_end'):
            return
        self._enqueue(('pi_event', {
            'segment': segment,
            'utc_ns': utc_ns,
            'row': row,
        }))

    def record_worker_event(self, *, segment: int, phase: str, order: int, operation: str,
                            engine: str, status: str | None = None, evidence: str | None = None,
                            build: str | None = None, node_ms: int | None = None,
                            pair_index: int | None = None):
        self._enqueue(('worker_event', {
            'segment': segment,
            'kind': f'worker_{phase}',
            'utc_ns': time_ns(),
            'order': order,
            'operation': operation,
            'engine': engine,
            'status': status,
            'evidence': evidence,
            'build': build,
            'node_ms': node_ms,
            'pair': pair_index,
        }))

    def flush(self, timeout: float = 5.0) -> bool:
        if self._closed:
            return True
        done = Event()
        if not self._enqueue(('flush', done)):
            return False
        return done.wait(timeout)

    def close(self, timeout: float = 5.0):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except Full:
            pass
        self._thread.join(timeout=timeout)
