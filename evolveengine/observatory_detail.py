"""Run-scoped, allowlisted detail views (trace, diffs, waterfall, research, harness, context).

Every read enforces Holdout exclusion, symlink-free path confinement, per-run
membership validation (including content-addressed blobs), and defense-in-depth
secret redaction.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
import difflib
import json
from pathlib import Path
import re
import sqlite3

from . import observatory_data as proj
from .v3_level3_trace import redact_obj

HEX64 = re.compile(r'^[0-9a-f]{64}\Z')
MAX_DIFF_LINES = 2000
MAX_INLINE_BLOB = 24 * 1024


def _ensure_eligible(root: Path):
    ok, reg, settlement = proj.is_eligible_run(root)
    if not ok:
        raise PermissionError('run is not eligible for Observatory detail access')
    return reg, settlement


def _utc_ns_from_iso(value: str | None) -> int | None:
    ms = proj.ts(value)
    return round(ms * 1_000_000) if ms is not None else None


def _run_origin_utc_ns(root: Path) -> int | None:
    try:
        meta = proj.read_json(proj.safe(root, 'trace', 'run-meta.json'))
    except ValueError:
        meta = {}
    if (v := proj.num(meta.get('started_utc_ns'))) is not None:
        return int(v)
    # Fallback to earliest model request timestamp if trace/run-meta.json absent
    dirs = sorted((p for p in root.glob('model-segment-*') if re.fullmatch(r'model-segment-\d{4,6}', p.name)),
                  key=lambda p: int(p.name.split('-')[-1]))
    for d in dirs:
        try:
            d = proj.safe(root, d.name)
            start = proj.read_json(proj.safe(root, d.name, '0001-start.json'))
        except ValueError:
            continue
        if (ns := _utc_ns_from_iso(start.get('started_utc'))) is not None:
            return ns
    return None


def _read_trace_events(root: Path) -> tuple[list[dict], set[str]]:
    """Load segment-*.jsonl events and collect the exact set of referenced blob hashes."""
    events: list[dict] = []
    referenced_blobs: set[str] = set()
    try:
        trace_dir = proj.safe(root, 'trace')
    except ValueError:
        return events, referenced_blobs
    if not trace_dir.is_dir():
        return events, referenced_blobs

    def add_blob(h):
        if isinstance(h, str) and HEX64.fullmatch(h):
            referenced_blobs.add(h)

    for seg_file in sorted(trace_dir.glob('segment-*.jsonl')):
        if not re.fullmatch(r'segment-\d{4,6}\.jsonl', seg_file.name):
            continue
        try:
            raw_lines = seg_file.read_bytes().splitlines()
        except OSError:
            continue
        for raw in raw_lines:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if not isinstance(rec, dict):
                continue
            events.append(rec)
            add_blob(rec.get('response_blob'))
            add_blob(rec.get('args_blob'))
            add_blob(rec.get('result_blob'))
            add_blob(rec.get('message_blob'))
            for m in rec.get('message_blobs') or ():
                if isinstance(m, dict):
                    add_blob(m.get('blob'))
    events.sort(key=lambda r: (r.get('utc_ns') or 0, r.get('segment') or 0, r.get('seq') or 0))
    return events, referenced_blobs


def _read_blob_raw(root: Path, blob_sha: str, cap: int = 4 << 20):
    path = proj.safe(root, 'trace', 'blobs', f'{blob_sha}.json')
    if not path.is_file():
        raise LookupError('blob not found')
    with path.open('rb') as handle:
        data = handle.read(cap + 1)
    if len(data) > cap:
        raise LookupError('blob exceeds display cap')
    return redact_obj(json.loads(data))


def read_run_blob(root: Path, blob_sha: str) -> dict:
    """Serve a blob ONLY if it is explicitly referenced by this eligible run."""
    _ensure_eligible(root)
    if not isinstance(blob_sha, str) or not HEX64.fullmatch(blob_sha):
        raise LookupError('invalid blob digest')
    _, referenced = _read_trace_events(root)
    if blob_sha not in referenced:
        raise LookupError('blob does not belong to this run')
    return {'id': blob_sha, 'content': _read_blob_raw(root, blob_sha)}


def _inline_small_blob(root: Path, blob_sha: str | None):
    if not isinstance(blob_sha, str) or not HEX64.fullmatch(blob_sha):
        return None
    try:
        path = proj.safe(root, 'trace', 'blobs', f'{blob_sha}.json')
        if path.is_file() and path.stat().st_size <= MAX_INLINE_BLOB:
            return redact_obj(json.loads(path.read_bytes()))
    except (OSError, ValueError, RecursionError):
        return None
    return None


def _group_turns(root: Path, events: list[dict]) -> list[dict]:
    """Assemble conversation turns with explicit ID correlation (toolCallId/call_sha256 -> engine)."""
    turns: list[dict] = []
    seg_turns: dict[int, list[dict]] = {}
    call_to_turn: dict[str, dict] = {}
    open_tools: dict[tuple[int, str], tuple[dict, dict]] = {}
    orphan_added = False

    def ensure_orphan_turn(seg: int, utc_ns: int | None, elapsed_ms: int | None) -> dict:
        nonlocal orphan_added
        orphan_added = True
        t = {
            'id': f'turn/{seg}/orphan-{len(turns) + 1}',
            'segment': seg,
            'attempt': None,
            'utc_ns': utc_ns,
            'elapsed_ms': elapsed_ms,
            'model': None,
            'response_blob': None,
            'response_inline': None,
            'tools': [],
            'worker_events': [],
            'engines': [],
            'has_error': False,
            'is_candidate_turn': False,
        }
        turns.append(t)
        return t

    def nearest_turn(seg: int, utc_ns: int | None) -> dict | None:
        candidates = seg_turns.get(seg)
        if not candidates:
            return None
        if utc_ns is None:
            return candidates[-1]
        chosen = None
        for t in candidates:
            if (t.get('utc_ns') or 0) <= utc_ns:
                chosen = t
            else:
                break
        return chosen

    # Pass 1: materialize all model_exchange turns and index call_sha256 -> turn
    # before correlating Pi tool events, so clock/settlement jitter between Pi and
    # Broker never splits a matched tool call into spurious orphan turns.
    for ev in events:
        if ev.get('kind') != 'model_exchange':
            continue
        seg = int(ev.get('segment') or 0)
        attempt = ev.get('attempt')
        resp_blob = ev.get('response_blob')
        turn = {
            'id': f'turn/{seg}/{attempt}',
            'segment': seg,
            'attempt': attempt,
            'utc_ns': ev.get('utc_ns'),
            'elapsed_ms': ev.get('elapsed_ms'),
            'model': {
                'started_utc': ev.get('started_utc'),
                'ended_utc': ev.get('ended_utc'),
                'wall_ms': ev.get('wall_ms'),
                'finish': ev.get('finish'),
                'result': ev.get('result'),
                'usage_status': ev.get('usage_status'),
                'input_tokens': ev.get('input_tokens'),
                'output_tokens': ev.get('output_tokens'),
                'reasoning_tokens': ev.get('reasoning_tokens'),
                'cache_hit_tokens': ev.get('cache_hit_tokens'),
                'cache_miss_tokens': ev.get('cache_miss_tokens'),
                'first_delta_ms': ev.get('first_delta_ms'),
                'last_delta_ms': ev.get('last_delta_ms'),
                'output_span_ms': ev.get('output_span_ms'),
                'request_message_count': ev.get('request_message_count'),
                'reasoning_chars': ev.get('reasoning_chars') or 0,
                'content_chars': ev.get('content_chars') or 0,
                'content_preview': ev.get('content_preview') or '',
                'rejection': ev.get('rejection'),
            },
            'response_blob': resp_blob,
            'response_inline': _inline_small_blob(root, resp_blob),
            'tools': [],
            'worker_events': [],
            'engines': [],
            'has_error': bool(ev.get('rejection') or ev.get('finish') == 'length' or ev.get('usage_status') == 'usage_uncertain'),
            'is_candidate_turn': False,
        }
        turns.append(turn)
        seg_turns.setdefault(seg, []).append(turn)
        for csha in ev.get('call_sha256s') or ():
            if isinstance(csha, str):
                call_to_turn[csha] = turn

    # Pass 2: correlate Pi tool starts/ends and Worker events to their owning turn.
    for ev in events:
        kind = ev.get('kind')
        seg = int(ev.get('segment') or 0)
        if kind == 'pi_tool_start':
            csha = ev.get('call_sha256')
            owner = call_to_turn.get(csha) if isinstance(csha, str) else None
            if owner is None:
                owner = nearest_turn(seg, ev.get('utc_ns')) or ensure_orphan_turn(seg, ev.get('utc_ns'), ev.get('elapsed_ms'))
            entry = {
                'tool': ev.get('tool'),
                'rpc_operation': ev.get('rpc_operation'),
                'tool_call_id': ev.get('tool_call_id'),
                'call_sha256': csha,
                'path': ev.get('path'),
                'paths': ev.get('paths'),
                'engine': ev.get('engine'),
                'parent': ev.get('parent'),
                'build': None,
                'evidence': None,
                'start_utc_ns': ev.get('utc_ns'),
                'end_utc_ns': None,
                'elapsed_ms': ev.get('elapsed_ms'),
                'duration_ms': None,
                'is_error': False,
                'args_preview': ev.get('args_preview') or '',
                'args_blob': ev.get('args_blob'),
                'args_inline': _inline_small_blob(root, ev.get('args_blob')),
                'result_preview': '',
                'result_blob': None,
                'result_inline': None,
                'rpc_status': None,
                'op_status': None,
            }
            owner['tools'].append(entry)
            if isinstance(csha, str):
                open_tools[(seg, csha)] = (entry, owner)
            if entry['engine'] and entry['engine'] not in owner['engines']:
                owner['engines'].append(entry['engine'])
            if entry['rpc_operation'] == 'candidate_submit':
                owner['is_candidate_turn'] = True
        elif kind == 'pi_tool_end':
            csha = ev.get('call_sha256')
            opened = open_tools.pop((seg, csha), None) if isinstance(csha, str) else None
            entry, opened_owner = opened if opened is not None else (None, None)
            owner = opened_owner or (call_to_turn.get(csha) if isinstance(csha, str) else None)
            if owner is None:
                owner = nearest_turn(seg, ev.get('utc_ns')) or ensure_orphan_turn(seg, ev.get('utc_ns'), ev.get('elapsed_ms'))
            if entry is None:
                entry = {
                    'tool': ev.get('tool'),
                    'rpc_operation': ev.get('rpc_operation'),
                    'tool_call_id': ev.get('tool_call_id'),
                    'call_sha256': csha,
                    'path': None,
                    'paths': None,
                    'engine': ev.get('engine'),
                    'parent': ev.get('parent'),
                    'build': ev.get('build'),
                    'evidence': ev.get('evidence'),
                    'start_utc_ns': ev.get('utc_ns'),
                    'end_utc_ns': ev.get('utc_ns'),
                    'elapsed_ms': ev.get('elapsed_ms'),
                    'duration_ms': ev.get('duration_ms'),
                    'is_error': bool(ev.get('is_error')),
                    'args_preview': '',
                    'args_blob': None,
                    'args_inline': None,
                    'result_preview': ev.get('result_preview') or '',
                    'result_blob': ev.get('result_blob'),
                    'result_inline': _inline_small_blob(root, ev.get('result_blob')),
                    'rpc_status': ev.get('rpc_status'),
                    'op_status': ev.get('op_status'),
                }
                owner['tools'].append(entry)
            else:
                entry['end_utc_ns'] = ev.get('utc_ns')
                entry['duration_ms'] = ev.get('duration_ms')
                entry['is_error'] = bool(ev.get('is_error'))
                entry['result_preview'] = ev.get('result_preview') or ''
                entry['result_blob'] = ev.get('result_blob')
                entry['result_inline'] = _inline_small_blob(root, ev.get('result_blob'))
                entry['rpc_status'] = ev.get('rpc_status')
                entry['op_status'] = ev.get('op_status')
                for k in ('engine', 'parent', 'build', 'evidence', 'rpc_operation'):
                    if ev.get(k) and not entry.get(k):
                        entry[k] = ev[k]
            if entry.get('is_error') or (isinstance(entry.get('rpc_status'), int) and entry['rpc_status'] >= 400):
                owner['has_error'] = True
            if entry.get('engine') and entry['engine'] not in owner['engines']:
                owner['engines'].append(entry['engine'])
            if entry.get('rpc_operation') == 'candidate_submit':
                owner['is_candidate_turn'] = True
        elif kind in ('worker_reserved', 'worker_settled', 'worker_uncertain'):
            owner = nearest_turn(seg, ev.get('utc_ns')) or (
                seg_turns[seg][-1] if seg_turns.get(seg) else ensure_orphan_turn(seg, ev.get('utc_ns'), ev.get('elapsed_ms'))
            )
            owner['worker_events'].append({
                'kind': kind,
                'operation': ev.get('operation'),
                'engine': ev.get('engine'),
                'status': ev.get('status'),
                'evidence': ev.get('evidence'),
                'build': ev.get('build'),
                'node_ms': ev.get('node_ms'),
                'pair': ev.get('pair'),
                'elapsed_ms': ev.get('elapsed_ms'),
            })
            if ev.get('engine') and ev['engine'] not in owner['engines']:
                owner['engines'].append(ev['engine'])
            if kind == 'worker_uncertain' or ev.get('status') in proj.FAILURES:
                owner['has_error'] = True
    if orphan_added:
        turns.sort(key=lambda t: (t.get('utc_ns') or 0, t.get('segment') or 0, t.get('attempt') or 0))
    return turns


def trace_view(root: Path, *, segment: int | None = None, tool: str | None = None,
               query: str | None = None, engine: str | None = None,
               errors_only: bool = False, only_candidates: bool = False,
               offset: int = 0, limit: int = 40) -> dict:
    _ensure_eligible(root)
    events, _ = _read_trace_events(root)
    has_content = bool(events)
    turns = _group_turns(root, events)
    segments = sorted({t['segment'] for t in turns})
    all_tools = sorted({
        (tc.get('rpc_operation') or tc.get('tool'))
        for t in turns for tc in t['tools']
        if (tc.get('rpc_operation') or tc.get('tool'))
    })

    q_lower = (query or '').strip().lower()
    filtered = []
    for t in turns:
        if segment is not None and t['segment'] != segment:
            continue
        if errors_only and not t['has_error']:
            continue
        if only_candidates and not t['is_candidate_turn']:
            continue
        if engine and engine not in t['engines']:
            continue
        if tool:
            if not any(tc.get('tool') == tool or tc.get('rpc_operation') == tool for tc in t['tools']):
                continue
        if q_lower:
            hay = [
                t['id'],
                (t.get('model') or {}).get('content_preview', ''),
                ' '.join(t['engines']),
            ]
            for tc in t['tools']:
                hay.extend([
                    str(tc.get('tool') or ''),
                    str(tc.get('rpc_operation') or ''),
                    str(tc.get('path') or ''),
                    ' '.join(tc.get('paths') or []),
                    str(tc.get('args_preview') or ''),
                    str(tc.get('result_preview') or ''),
                ])
            if q_lower not in ' '.join(hay).lower():
                continue
        filtered.append(t)

    offset = max(0, offset)
    limit = max(1, min(100, limit))
    return {
        'has_trace_content': has_content,
        'total_turns': len(turns),
        'matched_turns': len(filtered),
        'offset': offset,
        'limit': limit,
        'segments': segments,
        'tools': all_tools,
        'turns': filtered[offset:offset + limit],
    }


def _load_engine_objects(root: Path) -> tuple[dict[str, dict], dict[str, str | None], str | None]:
    """Read registered engine source objects and parent pointers from trajectory.sqlite3."""
    engines: dict[str, dict] = {}
    parents: dict[str, str | None] = {}
    baseline: str | None = None
    db_path = proj.safe(root, 'run', 'trajectory.sqlite3')
    if not db_path.is_file():
        return engines, parents, baseline
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True, timeout=0.7)) as db:
        db.execute('PRAGMA query_only=ON')
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone() is None:
            return engines, parents, baseline
        for _, raw in db.execute('SELECT id, content FROM events ORDER BY ordinal'):
            try:
                row = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if not isinstance(row, dict):
                continue
            op = row.get('operation')
            if op == 'engine_registered':
                eng = proj.pin(row.get('engine'))
                src_obj = proj.pin(row.get('source_object'))
                if eng:
                    parents[eng] = proj.pin(row.get('parent'))
                    if baseline is None and parents[eng] is None:
                        baseline = eng
                    if src_obj and eng not in engines:
                        saved = db.execute('SELECT content FROM objects WHERE id=? AND kind=?',
                                           (src_obj, 'level3_engine')).fetchone()
                        if saved:
                            try:
                                engines[eng] = json.loads(saved[0])
                            except (ValueError, RecursionError):
                                pass
            elif op == 'development_reference':
                if eng := proj.pin(row.get('engine')):
                    baseline = eng
    return engines, parents, baseline


def _compute_file_diffs(base_args: dict | None, target_args: dict | None) -> dict:
    base_files = (base_args or {}).get('files') or {}
    target_files = (target_args or {}).get('files') or {}
    base_del = set((base_args or {}).get('deleted') or [])
    target_del = set((target_args or {}).get('deleted') or [])
    all_paths = sorted(set(base_files) | set(target_files) | base_del | target_del)
    file_diffs = []
    runtime_diff = None

    for path in all_paths:
        a = base_files.get(path)
        b = target_files.get(path)
        if a == b and (path in base_del) == (path in target_del):
            continue
        if path in target_del and path not in base_del:
            status = 'deleted'
        elif a is None and b is not None:
            status = 'added'
        else:
            status = 'modified'
        a_lines = (a or '').splitlines(keepends=True)
        b_lines = (b or '').splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(a_lines, b_lines, fromfile=f'a/{path}', tofile=f'b/{path}', n=3))
        truncated = len(diff_lines) > MAX_DIFF_LINES
        if truncated:
            diff_lines = diff_lines[:MAX_DIFF_LINES]
        added_cnt = sum(1 for ln in diff_lines if ln.startswith('+') and not ln.startswith('+++'))
        removed_cnt = sum(1 for ln in diff_lines if ln.startswith('-') and not ln.startswith('---'))
        file_diffs.append({
            'path': path,
            'status': status,
            'added_lines': added_cnt,
            'removed_lines': removed_cnt,
            'truncated': truncated,
            'diff': ''.join(diff_lines),
        })
        if path == 'ee_runtime.json':
            try:
                old_rt = json.loads(a) if a else {}
                new_rt = json.loads(b) if b else {}
                if isinstance(old_rt, dict) and isinstance(new_rt, dict):
                    keys = sorted(set(old_rt) | set(new_rt))
                    runtime_diff = [
                        {'key': k, 'before': old_rt.get(k), 'after': new_rt.get(k)}
                        for k in keys if old_rt.get(k) != new_rt.get(k)
                    ]
            except (ValueError, RecursionError):
                pass

    base_build = (base_args or {}).get('build_args') or {}
    target_build = (target_args or {}).get('build_args') or {}
    build_diff = [
        {'key': k, 'before': base_build.get(k), 'after': target_build.get(k)}
        for k in sorted(set(base_build) | set(target_build))
        if base_build.get(k) != target_build.get(k)
    ]
    return {
        'files': file_diffs,
        'runtime_diff': runtime_diff,
        'build_args_diff': build_diff,
    }


def candidate_detail(root: Path, run_id: str, engine_sha: str) -> dict:
    """Return verified candidate diff, build/quality/measure receipts, and ID-linked turns."""
    _ensure_eligible(root)
    if not isinstance(engine_sha, str) or not HEX64.fullmatch(engine_sha):
        raise LookupError('invalid engine digest')
    engines, parents, baseline = _load_engine_objects(root)
    if engine_sha not in parents and engine_sha not in engines:
        raise LookupError('engine not registered in this run')

    snap = proj.local_run(root, run_id)
    parent = parents.get(engine_sha)
    target_args = engines.get(engine_sha)
    parent_args = engines.get(parent) if parent else None
    e0_args = engines.get(baseline) if baseline else None

    diff_vs_parent = _compute_file_diffs(parent_args, target_args) if parent else _compute_file_diffs(None, target_args)
    diff_vs_e0 = _compute_file_diffs(e0_args, target_args) if baseline and baseline != engine_sha else _compute_file_diffs(None, None)

    worker_events = [e for e in snap['events'] if e.get('engine') == engine_sha and e.get('kind') in ('worker_settled', 'worker_uncertain')]
    builds = [e for e in worker_events if e.get('operation') == 'develop_build']
    qualities = [e for e in worker_events if e.get('operation') == 'develop_quality']
    points = [p for p in snap['points'] if p.get('engine') == engine_sha]
    summaries = [s for s in snap.get('build_summaries', []) if s.get('engine') == engine_sha]
    version = next((v for v in snap['versions'] if v['id'] == engine_sha), None)

    # ID-correlated conversation slice:
    # 1. Find exact turn(s) whose candidate_submit tool returned engine == engine_sha
    # 2. Include preceding turns back to prior candidate_submit in same segment (cap 20)
    # 3. Include subsequent turns referencing engine == engine_sha
    trace_events, _ = _read_trace_events(root)
    all_turns = _group_turns(root, trace_events)
    correlated = []
    seen_ids = set()
    submit_indices = [
        i for i, t in enumerate(all_turns)
        if any(tc.get('rpc_operation') == 'candidate_submit' and tc.get('engine') == engine_sha for tc in t['tools'])
    ]
    if submit_indices:
        first_idx = submit_indices[0]
        seg = all_turns[first_idx]['segment']
        start_idx = first_idx
        while start_idx > 0 and all_turns[start_idx - 1]['segment'] == seg and (first_idx - start_idx) < 20:
            prev = all_turns[start_idx - 1]
            if any(tc.get('rpc_operation') == 'candidate_submit' and tc.get('engine') and tc.get('engine') != engine_sha for tc in prev['tools']):
                break
            start_idx -= 1
        for i in range(start_idx, first_idx + 1):
            t = all_turns[i]
            role = 'submitted_candidate' if i == first_idx else 'preceding_work'
            correlated.append({**t, 'correlation': role})
            seen_ids.add(t['id'])
    for t in all_turns:
        if t['id'] not in seen_ids and engine_sha in t['engines']:
            correlated.append({**t, 'correlation': 'evaluated_candidate'})
            seen_ids.add(t['id'])

    return redact_obj({
        'engine': engine_sha,
        'iteration': version['iteration'] if version else None,
        'parent': parent,
        'baseline': baseline,
        'scoring': snap.get('scoring'),
        'has_trace_content': bool(trace_events),
        'diff_vs_parent': diff_vs_parent,
        'diff_vs_e0': diff_vs_e0,
        'builds': builds,
        'qualities': qualities,
        'points': points,
        'build_summaries': summaries,
        'conversation': correlated,
    })


def waterfall_view(root: Path, run_id: str) -> dict:
    """Build live & historical time waterfall spans across model, tools, research, and Worker."""
    _ensure_eligible(root)
    origin_ns = _run_origin_utc_ns(root)
    spans = []
    totals = {'model_ttft_ms': 0.0, 'model_stream_ms': 0.0, 'pi_tool_ms': 0.0,
              'research_ms': 0.0, 'build_ms': 0.0, 'quality_ms': 0.0, 'measure_ms': 0.0, 'other_worker_ms': 0.0}

    def rel_ms(utc_ns: int | None) -> float | None:
        nonlocal origin_ns
        if utc_ns is None:
            return None
        if origin_ns is None:
            origin_ns = utc_ns
        return max(0.0, (utc_ns - origin_ns) / 1e6)

    # 1. Model requests (always available from model-segment-*/NNNN-result.json)
    dirs = sorted((p for p in root.glob('model-segment-*') if re.fullmatch(r'model-segment-\d{4,6}', p.name)),
                  key=lambda p: int(p.name.split('-')[-1]))
    for d in dirs:
        try:
            d = proj.safe(root, d.name)
        except ValueError:
            continue
        seg = int(d.name.split('-')[-1])
        for entry in sorted(d.glob('*-result.json')):
            rec = proj.read_json(entry)
            usage = proj.mapping(rec.get('usage'))
            s_ns = _utc_ns_from_iso(rec.get('started_utc'))
            e_ns = _utc_ns_from_iso(rec.get('ended_utc'))
            if s_ns is None or e_ns is None or e_ns <= s_ns:
                continue
            s_ms = rel_ms(s_ns)
            e_ms = rel_ms(e_ns)
            assert s_ms is not None and e_ms is not None
            gen = proj.mapping(usage.get('generation_timing'))
            fd = proj.num(gen.get('first_delta_ms'))
            attempt = rec.get('attempt')
            if fd is not None and fd > 0 and s_ms + fd < e_ms:
                ttft_end = min(e_ms, s_ms + fd)
                spans.append({'lane': 'Model TTFT', 'label': f'S{seg} #{attempt} wait', 'start_ms': s_ms, 'end_ms': ttft_end,
                              'duration_ms': ttft_end - s_ms, 'segment': seg, 'status': usage.get('finish') or 'ok'})
                spans.append({'lane': 'Model Stream', 'label': f'S{seg} #{attempt} stream', 'start_ms': ttft_end, 'end_ms': e_ms,
                              'duration_ms': e_ms - ttft_end, 'segment': seg, 'status': usage.get('finish') or 'ok'})
                totals['model_ttft_ms'] += ttft_end - s_ms
                totals['model_stream_ms'] += e_ms - ttft_end
            else:
                spans.append({'lane': 'Model Stream', 'label': f'S{seg} #{attempt}', 'start_ms': s_ms, 'end_ms': e_ms,
                              'duration_ms': e_ms - s_ms, 'segment': seg, 'status': usage.get('finish') or 'ok'})
                totals['model_stream_ms'] += e_ms - s_ms

    # 2. Pi tools (prefer live trace, fallback to result.json)
    trace_events, _ = _read_trace_events(root)
    if trace_events:
        opened = {}
        for ev in trace_events:
            kind = ev.get('kind')
            seg = int(ev.get('segment') or 0)
            csha = ev.get('call_sha256')
            if kind == 'pi_tool_start' and csha:
                opened[(seg, csha)] = ev
            elif kind == 'pi_tool_end' and csha and (st := opened.pop((seg, csha), None)):
                s_ms = rel_ms(st.get('utc_ns'))
                e_ms = rel_ms(ev.get('utc_ns'))
                if s_ms is not None and e_ms is not None and e_ms >= s_ms:
                    tool = ev.get('rpc_operation') or ev.get('tool') or 'tool'
                    dur = max(1.0, e_ms - s_ms)
                    spans.append({'lane': 'Pi Tools', 'label': f"{tool} {st.get('args_preview', '')[:48]}".strip(),
                                  'start_ms': s_ms, 'end_ms': s_ms + dur, 'duration_ms': dur, 'segment': seg,
                                  'status': 'error' if ev.get('is_error') else 'ok', 'engine': ev.get('engine')})
                    totals['pi_tool_ms'] += dur
    else:
        terminal = proj.read_json(proj.safe(root, 'run', 'result.json'), cap=16 << 20)
        for seg_row in terminal.get('segments') or ():
            if not isinstance(seg_row, dict):
                continue
            seg = int(seg_row.get('segment') or 0)
            opened = {}
            for ev in proj.mapping(seg_row.get('report')).get('pi_tool_events') or ():
                if not isinstance(ev, dict):
                    continue
                csha = ev.get('call_sha256')
                if ev.get('event') == 'tool_execution_start' and csha:
                    opened[csha] = ev
                elif ev.get('event') == 'tool_execution_end' and csha and (st := opened.pop(csha, None)):
                    s_ms = rel_ms(st.get('utc_ns'))
                    e_ms = rel_ms(ev.get('utc_ns'))
                    if s_ms is not None and e_ms is not None and e_ms >= s_ms:
                        dur = max(1.0, e_ms - s_ms)
                        spans.append({'lane': 'Pi Tools', 'label': str(ev.get('tool') or 'tool'),
                                      'start_ms': s_ms, 'end_ms': s_ms + dur, 'duration_ms': dur, 'segment': seg, 'status': 'ok'})
                        totals['pi_tool_ms'] += dur

    # 3. Web Research calls
    wr_dir = root / 'web-research'
    if wr_dir.is_dir():
        for entry in sorted(wr_dir.glob('*-settlement.json')):
            call = entry.name.split('-', 1)[0]
            req = proj.read_json(wr_dir / f'{call}-request.json')
            settle = proj.read_json(entry)
            s_ns = _utc_ns_from_iso(req.get('utc'))
            e_ns = _utc_ns_from_iso(settle.get('ended_utc'))
            s_ms, e_ms = rel_ms(s_ns), rel_ms(e_ns)
            if s_ms is not None and e_ms is not None and e_ms >= s_ms:
                dur = max(1.0, e_ms - s_ms)
                spans.append({'lane': 'Research', 'label': f"{settle.get('operation', 'research')} #{call}",
                              'start_ms': s_ms, 'end_ms': s_ms + dur, 'duration_ms': dur,
                              'status': 'ok' if settle.get('status') == 200 else 'error'})
                totals['research_ms'] += dur

    # 4. Worker operations (elapsed_ms relative to launch start)
    wo_dir = root / 'worker-operations'
    if wo_dir.is_dir():
        for res_file in sorted(wo_dir.glob('*-reserved.json')):
            prefix = res_file.name.split('-', 1)[0]
            start = proj.read_json(res_file)
            end = proj.read_json(wo_dir / f'{prefix}-transport.json')
            s_ms = proj.num(start.get('elapsed_ms'))
            e_ms = proj.num(end.get('elapsed_ms'))
            if s_ms is None or e_ms is None or e_ms < s_ms:
                continue
            op = str(start.get('operation') or 'worker')
            dur = max(1.0, e_ms - s_ms)
            lane = {
                'develop_build': 'Worker Build',
                'develop_quality': 'Worker Quality',
                'develop_measure': 'Worker Measure',
            }.get(op, 'Worker Other')
            spans.append({'lane': lane, 'label': f"{op.removeprefix('develop_')} {(start.get('engine') or '')[:10]}",
                          'start_ms': s_ms, 'end_ms': s_ms + dur, 'duration_ms': dur,
                          'engine': start.get('engine'), 'evidence': end.get('evidence'),
                          'status': end.get('settlement') or 'ok'})
            if op == 'develop_build':
                totals['build_ms'] += dur
            elif op == 'develop_quality':
                totals['quality_ms'] += dur
            elif op == 'develop_measure':
                totals['measure_ms'] += dur
            else:
                totals['other_worker_ms'] += dur

    spans.sort(key=lambda sp: (sp['start_ms'], sp['lane']))
    total_wall = max((sp['end_ms'] for sp in spans), default=0.0)
    return {
        'total_wall_ms': total_wall,
        'breakdown': totals,
        'lanes': ['Model TTFT', 'Model Stream', 'Pi Tools', 'Research', 'Worker Build', 'Worker Quality', 'Worker Measure', 'Worker Other'],
        'spans': spans[:5000],
    }


def research_view(root: Path) -> dict:
    """Project web-research queries, results, fetched pages, and research_reuse records."""
    _ensure_eligible(root)
    try:
        wr_dir = proj.safe(root, 'web-research')
    except ValueError:
        wr_dir = None
    if not wr_dir or not wr_dir.is_dir():
        return {'available': False, 'searches': [], 'fetches': [], 'pages': [], 'reuses': []}

    searches, fetches, pages, reuses = [], [], [], []
    engines, _, _ = _load_engine_objects(root)

    for req_file in sorted(wr_dir.glob('*-request.json')):
        call = req_file.name.split('-', 1)[0]
        req = proj.read_json(req_file)
        resp = proj.read_json(wr_dir / f'{call}-response.json', cap=16 << 20)
        settle = proj.read_json(wr_dir / f'{call}-settlement.json')
        op = req.get('operation')
        payload = proj.mapping(req.get('payload'))
        if op == 'search':
            results = []
            for row in (resp.get('results') or [])[:50]:
                if isinstance(row, dict):
                    results.append({k: row.get(k) for k in ('title', 'url', 'snippet', 'date', 'site_name')})
            searches.append({
                'call': int(call) if call.isdigit() else call,
                'utc': req.get('utc'),
                'query': payload.get('query'),
                'count': payload.get('count'),
                'status': settle.get('status'),
                'elapsed_ms': settle.get('elapsed_ms'),
                'results': results,
            })
        elif op == 'fetch':
            content = resp.get('content') if isinstance(resp.get('content'), str) else ''
            fetches.append({
                'call': int(call) if call.isdigit() else call,
                'utc': req.get('utc'),
                'url': payload.get('url'),
                'title': resp.get('title'),
                'status': settle.get('status'),
                'elapsed_ms': settle.get('elapsed_ms'),
                'content_bytes': len(content.encode('utf-8')),
            })

    for page_file in sorted(wr_dir.glob('page-*.json')):
        pid = page_file.stem.removeprefix('page-')
        if not HEX64.fullmatch(pid):
            continue
        doc = proj.read_json(page_file, cap=32 << 20)
        body = doc.get('content') if isinstance(doc.get('content'), str) else ''
        pages.append({
            'id': pid,
            'url': doc.get('url'),
            'title': doc.get('title'),
            'bytes': len(body.encode('utf-8')),
            'preview': body[:320],
        })

    for reuse_file in sorted(wr_dir.glob('reuse-*.json')):
        rid = reuse_file.stem.removeprefix('reuse-')
        if not HEX64.fullmatch(rid):
            continue
        doc = proj.read_json(reuse_file)
        paths = [str(p) for p in (doc.get('paths') or [])]
        matched_engines = [
            eng for eng, args in engines.items()
            if any(p in ((args or {}).get('files') or {}) for p in paths)
        ]
        reuses.append({
            'receipt': rid,
            'kind': doc.get('kind'),
            'paths': paths,
            'url': doc.get('url'),
            'revision': doc.get('revision'),
            'license': doc.get('license'),
            'identity': doc.get('identity'),
            'matched_engines': matched_engines,
        })

    return redact_obj({'available': True, 'searches': searches, 'fetches': fetches, 'pages': pages, 'reuses': reuses})


def read_research_page(root: Path, page_id: str) -> dict:
    _ensure_eligible(root)
    if not isinstance(page_id, str) or not HEX64.fullmatch(page_id):
        raise LookupError('invalid page digest')
    path = proj.safe(root, 'web-research', f'page-{page_id}.json')
    if not path.is_file():
        raise LookupError('research page not found')
    doc = proj.read_json(path, cap=8 << 20)
    body = doc.get('content') if isinstance(doc.get('content'), str) else ''
    truncated = len(body) > 200_000
    return redact_obj({
        'id': page_id,
        'url': doc.get('url'),
        'title': doc.get('title'),
        'bytes': len(body.encode('utf-8')),
        'truncated': truncated,
        'content': body[:200_000],
    })


def harness_view(root: Path, run_id: str, *, left: str | None = None, right: str | None = None) -> dict:
    """Return Harness versions, 4-stage evidence matrix, and file diffs."""
    _ensure_eligible(root)
    snap = proj.local_run(root, run_id)
    try:
        lineage_root = proj.safe(root, 'run', 'harness-lineage', 'versions')
    except ValueError:
        lineage_root = None

    submitted = {e.get('harness') for e in snap['events'] if e.get('kind') == 'harness_submit' and e.get('harness')}
    activated = {e.get('harness') for e in snap['events'] if e.get('kind') == 'harness_activate' and e.get('harness')}
    attached_by_h: dict[str, list[int]] = {}
    for e in snap['events']:
        if e.get('kind') == 'session_harness_attached' and e.get('harness'):
            attached_by_h.setdefault(e['harness'], []).append(int(e.get('session') or 0))

    # Count verified prompt_fragments loaded across model requests
    loaded_fragment_turns: dict[str, int] = {}
    for d in sorted(root.glob('model-segment-*')):
        if not re.fullmatch(r'model-segment-\d{4,6}', d.name):
            continue
        for entry in d.glob('*-result.json'):
            rec = proj.read_json(entry)
            frags = proj.mapping(rec.get('usage')).get('prompt_fragments') or []
            for fh in frags:
                if isinstance(fh, str) and HEX64.fullmatch(fh):
                    loaded_fragment_turns[fh] = loaded_fragment_turns.get(fh, 0) + 1

    versions = []
    version_files: dict[str, dict[str, str]] = {}
    if lineage_root and lineage_root.is_dir():
        for folder in sorted(lineage_root.iterdir()):
            vid = folder.name
            if not folder.is_dir() or not HEX64.fullmatch(vid):
                continue
            manifest = proj.read_json(folder / 'manifest.json')
            load_cfg = proj.mapping(manifest.get('load'))
            files_meta = proj.mapping(manifest.get('files'))
            files_dir = folder / 'files'
            contents = {}
            if files_dir.is_dir():
                for fp in sorted(files_dir.rglob('*')):
                    if fp.is_file() and not fp.is_symlink():
                        rel = fp.relative_to(files_dir).as_posix()
                        try:
                            contents[rel] = fp.read_text(encoding='utf-8', errors='replace')
                        except OSError:
                            pass
            version_files[vid] = contents
            prompt_paths = list(load_cfg.get('prompts') or []) + list(load_cfg.get('memory') or [])
            prompt_hashes = [proj.mapping(files_meta.get(p)).get('sha256') for p in prompt_paths]
            prompt_hits = sum(loaded_fragment_turns.get(h, 0) for h in prompt_hashes if h)
            versions.append({
                'id': vid,
                'files': sorted(contents.keys()),
                'load': load_cfg,
                'evidence': {
                    'submitted': vid in submitted,
                    'activated': vid in activated or vid == snap.get('active_harness'),
                    'attached_sessions': sorted(set(attached_by_h.get(vid, []))),
                    'prompt_loaded_turns': prompt_hits,
                    'invoked_tools': 0,  # Populated below if trace has extension tool calls
                },
            })

    # Compute diff between left & right (default: first/prev vs active)
    ids = [v['id'] for v in versions]
    if not right:
        right = snap.get('active_harness') if snap.get('active_harness') in version_files else (ids[-1] if ids else None)
    if not left and len(ids) >= 2:
        left = ids[0] if ids[0] != right else ids[-2]
    diff = None
    if right in version_files:
        left_files = version_files.get(left, {}) if left else {}
        right_files = version_files[right]
        diff = _compute_file_diffs({'files': left_files}, {'files': right_files})['files']

    return redact_obj({
        'active_harness': snap.get('active_harness'),
        'loaded_harness': snap.get('loaded_harness'),
        'versions': versions,
        'diff_left': left,
        'diff_right': right,
        'diff': diff or [],
    })


def context_growth_view(root: Path) -> dict:
    """Return per-request context growth and cache hit/miss time series."""
    _ensure_eligible(root)
    points = []
    idx = 0
    dirs = sorted((p for p in root.glob('model-segment-*') if re.fullmatch(r'model-segment-\d{4,6}', p.name)),
                  key=lambda p: int(p.name.split('-')[-1]))
    for d in dirs:
        try:
            d = proj.safe(root, d.name)
        except ValueError:
            continue
        seg = int(d.name.split('-')[-1])
        for entry in sorted(d.glob('*-result.json')):
            rec = proj.read_json(entry)
            u = proj.mapping(rec.get('usage'))
            if not u:
                continue
            idx += 1
            cache = proj.mapping(u.get('reported_cache_usage'))
            gen = proj.mapping(u.get('generation_timing'))
            inp = proj.num(u.get('input_tokens'))
            hit = proj.num(cache.get('hit'))
            miss = proj.num(cache.get('miss'))
            points.append({
                'index': idx,
                'segment': seg,
                'attempt': rec.get('attempt'),
                'at_ms': proj.ts(rec.get('started_utc')),
                'input_tokens': inp,
                'cache_hit_tokens': hit,
                'cache_miss_tokens': miss,
                'cache_hit_rate': (hit / inp) if (inp and hit is not None and inp > 0) else None,
                'output_tokens': proj.num(u.get('output_including_reasoning')),
                'reasoning_tokens': proj.num(u.get('reasoning_tokens')),
                'wall_ms': proj.num(u.get('wall_ms')),
                'first_delta_ms': proj.num(gen.get('first_delta_ms')),
                'finish': u.get('finish'),
            })
    return {'points': points}
