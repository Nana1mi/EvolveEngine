"""Allowlisted projection of trusted controller receipts for RSI Observatory."""
from datetime import datetime
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
import sqlite3
from contextlib import closing

from .schema import canonical, digest as object_digest
from .v3_level3_scoring import weighted_score, weights_for

SCENES = ('W1', 'W2', 'W3', 'W4')
METRICS = ('output_toks_per_s', 'tpot_ms', 'ttft_ms', 'e2e_ms', 'dev_score',
           'share_ge_40_tps', 'share_ge_50_tps', 'p95_itl_ms', 'unfinished_count')
WORKER_OPS = frozenset('develop_build develop_quality develop_measure develop_profile develop_diagnostic'.split())
STATUSES = frozenset('''built valid observed violation insufficient_evidence build_failed
    cleanup_uncertain timeout timed_out uncertain completed_non_scoring incomplete_non_scoring
    output_replay_mismatch timed_output_mismatch workload_drift insufficient_budget
    delivered incomplete stopped paused failed_or_interrupted operator_interrupted
    sampled non_scoring pending pending_review provisional_pending_weight_review
    candidate_server_reported_not_verified not_run pass fail active settled reserved
    confirmed rejected succeeded cancelled admitted ready registered imported started
    complete tool_calls stop length content_filter generation_truncated upstream_transport
    upstream_http_error usage_uncertain input_reservation_rejected request_rejected
    worker_transport agent_delivery runtime_failure process_exit launcher_failed
    exhausted consecutive_failures lease_overrun wall_clock resource_exhausted
    repeated_failure_paused infrastructure_repair worker_fault operator_pause operator_pause_after_segment unplanned_outage
    context_generation_window network_failure guest_failure worker_failure worker_uncertain'''.split())
FAILURES = STATUSES & frozenset('''violation insufficient_evidence build_failed cleanup_uncertain
    timeout timed_out uncertain output_replay_mismatch timed_output_mismatch workload_drift
    incomplete_non_scoring insufficient_budget generation_truncated length upstream_transport
    upstream_http_error usage_uncertain input_reservation_rejected request_rejected
    failed_or_interrupted operator_interrupted runtime_failure process_exit launcher_failed
    lease_overrun wall_clock resource_exhausted rejected repeated_failure_paused context_generation_window
    network_failure guest_failure worker_failure worker_uncertain'''.split())
REPLAY_CASES = frozenset('numeric_32 natural_eos_4 autoregressive_full_vocab mixed_tail_absolute paired_cancel_numeric'.split())
CASES = REPLAY_CASES | {'sampling_three_bin'}
TOOLS = frozenset('read write edit bash grep find ls develop_build develop_quality develop_measure develop_profile develop_diagnostic candidate_submit harness_submit harness_activate harness_asset_save artifact_query checkpoint_save research_search research_read research_fetch deliver handoff'.split())
CONTINUATIONS = frozenset(('initial', 'ordinary', 'recovery', 'stopped', 'delivered', 'nominated'))
PINS = re.compile(r'[0-9a-f]{64}\Z')
RUN_ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,47}\Z')
MAX_EVENTS, MAX_MODEL_ROWS = 25_000, 10_000
MAX_EPOCH_MS = 8_640_000_000_000_000  # ECMAScript Date domain.
BASE = '25003e6439f1aee05315b9b166c4bd4407b0e1ae2466ea6e653f4777981cab9f'


def mapping(value):
    return value if isinstance(value, dict) else {}


def pin(value):
    return value if isinstance(value, str) and PINS.fullmatch(value) else None


def status(value):
    return value if isinstance(value, str) and value in STATUSES else None


def allowed(value, whitelist):
    return value if isinstance(value, str) and value in whitelist else None


def num(value):
    # Decimal strings are used by the budget ledger. Reject arbitrary strings.
    if isinstance(value, str) and len(value) <= 36 and re.fullmatch(r'\d+(?:\.\d+)?', value):
        value = float(value)
    return value if type(value) in (int, float) and 0 <= value <= 10**20 else None


def ts(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)', value):
        return None
    try:
        return datetime.fromisoformat(value).timestamp() * 1000
    except (ValueError, OSError, OverflowError):
        return None


def safe(root, *parts):
    path = root.joinpath(*parts)
    # Refuse symlinks/junctions, including an alias inside the same root.
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents):
        raise ValueError('aliased data path')
    return path


def read_json(path, cap=4 << 20):
    try:
        with path.open('rb') as src:
            data = src.read(cap + 1)
        if len(data) > cap:
            return {}
        return mapping(json.loads(data))
    except (OSError, ValueError, RecursionError):
        # The controller may be halfway through writing an exclusive receipt.
        return {}


def metrics(value):
    v = mapping(value)
    return {s: {k: val for k in METRICS if (val := num(mapping(v.get(s)).get(k))) is not None}
            for s in SCENES if isinstance(v.get(s), dict)}


def ratios(value):
    v = mapping(value)
    return {s: x for s in SCENES if (x := num(v.get(s))) is not None}


def event(identity, kind, source='host', **kw):
    return dict(id=identity, kind=kind, source=source, **{k: v for k, v in kw.items() if v is not None})


def sha_bytes(blob):
    return sha256(blob).hexdigest()


def empty(run_id, label, mode):
    return dict(schema_version=1, id=run_id, label=label, mode=mode, status='awaiting_receipts',
                delivery=None, baseline=None, active_harness=None, loaded_harness=None, points=[], events=[],
                versions=[], resources={}, limits={}, notices=[], source_ref=None)


def archive_runs(path):
    payload = path.read_bytes()
    doc = json.loads(payload)
    if doc.get('schema') != 'spark26-level3-exploration-observation-v1':
        raise ValueError('wrong Observatory archive schema')
    result = {}
    for n, row in enumerate(doc['runs'], 1):
        run_id = f'archive-{n:02}'
        out = empty(run_id, f'Level 3 archive / {n:02}', 'archive')
        out.update(status=status(row.get('terminal_status')) or 'settlement_recorded',
                   stop_cause=status(row.get('stop_cause')), source_ref=sha_bytes(payload),
                   active_harness=pin(row.get('active_harness')), baseline=BASE)
        r, m = mapping(row.get('resources')), mapping(row.get('model'))
        out['resources'] = dict(wall_ms=times(r.get('wall_minutes'), 60000), node_ms=times(r.get('gpu_node_minutes'), 60000),
            ledger_rmb=num(r.get('local_api_worst_case_reservation_ledger_rmb')), input_tokens=num(m.get('input_tokens')),
            output_tokens=num(m.get('output_tokens_including_reasoning')), reasoning_tokens=num(m.get('reported_reasoning_tokens')),
            cache_hit_tokens=num(m.get('reported_cache_hit_tokens')), cache_miss_tokens=num(m.get('reported_cache_miss_tokens')),
            requests=num(m.get('settled_requests')), fee_idle=num(r.get('usage_fee_estimate_idle_rmb')),
            fee_peak=num(r.get('usage_fee_estimate_peak_rmb')), invoice_verified=False)
        seen = {}
        for reading in row.get('quality_passing_development_points', []):
            eng, ref = pin(reading.get('engine')), pin(reading.get('candidate_timing_evidence'))
            if eng is None or ref is None or ref in {p['id'] for p in out['points']}:
                continue
            identity = engine_version(out, seen, eng)
            index = len(out['points']) + 1
            out['points'].append(dict(id=ref, engine=eng, iteration=identity, index=index,
                 build=pin(reading.get('build')), pair=num(reading.get('measurement_pair')), reference=None,
                 aggregate=num(reading.get('equal_scene_geomean_vs_paired_e0')), aggregate_kind='equal_scene_geomean',
                 ratios=ratios(reading.get('scene_ratios')), scenes={}, reference_scenes={}, known_quality=True))
            out['events'].append(event('measurement/'+ref, 'comparison', 'verified', engine=eng, evidence=ref,
                                       point_id=ref, iteration=identity, point_index=index, status='observed'))
        delivered = pin(row.get('explicitly_delivered_engine'))
        if delivered:
            out['delivery'] = dict(engine=delivered, evidence=None, point_id=None, disposition='delivered')
            out['events'].append(event('run/delivery', 'delivery', 'agent', engine=delivered, status='delivered'))
        else:
            cause = status(row.get('stop_cause'))
            out['events'].append(event('run/settlement', 'run_settlement', status=out['status'], cause=cause,
                                       failure=cause in FAILURES))
        out['notices'] = ['Archive summary only: per-point clocks, failed candidates, tool trace and load evidence are unavailable.']
        result[run_id] = out
    return result


def times(value, mult):
    n = num(value)
    return n * mult if n is not None else None


def engine_version(out, seen, engine, parent=None):
    if engine is None:
        return None
    if engine not in seen:
        seen[engine] = len(seen)
        out['versions'].append(dict(id=engine, iteration=seen[engine], parent=parent))
    return seen[engine]


def journal(path):
    rows, objects = [], {}
    if not path.is_file():
        return rows, objects, False
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.7)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone() is None:
            return rows, objects, False
        for ordinal, key, raw in db.execute('SELECT ordinal,id,CASE WHEN length(content)<=? THEN content END FROM events ORDER BY ordinal LIMIT ?',
                                            (1 << 20, MAX_EVENTS+1)):
            if len(rows) == MAX_EVENTS:
                return rows, objects, True
            try:
                doc = mapping(json.loads(raw)) if raw is not None else {}
            except (ValueError, RecursionError):
                doc = {}
            rows.append((ordinal, key, doc))
            address = pin(doc.get('result_object')) if doc.get('operation') == 'worker_settled' else None
            if not address or address in objects:
                continue
            saved = db.execute('SELECT content FROM objects WHERE id=? AND kind=? AND length(content)<=?',
                               (address, 'level3_worker_result', 4 << 20)).fetchone()
            if saved:
                try:
                    row = mapping(json.loads(saved[0]))
                    if object_digest(dict(kind='level3_worker_result', content=row)) == address:
                        objects[address] = row
                except (ValueError, RecursionError):
                    pass
    return rows, objects, False


def worker_clocks(root, order, evidence=None):
    prefix = f'{order:05}'
    start = read_json(safe(root, 'worker-operations', prefix+'-reserved.json'))
    end = read_json(safe(root, 'worker-operations', prefix+'-transport.json'))
    return start, end if evidence is None or evidence == pin(end.get('evidence')) else {}


def bound_start(start, pending):
    _, engine, operation, reservation, pair = pending
    return (start if pin(start.get('engine')) == engine and start.get('operation') == operation
            and num(start.get('cap_ms')) == reservation and num(start.get('measurement_pair')) == pair else {})


def clock(start, end, node):
    ledger = mapping(start.get('api_cost_reservation_ledger'))
    return {k: v for k, v in dict(wall_ms=num(end.get('elapsed_ms')), dispatch_ms=num(start.get('elapsed_ms')),
        input_tokens=num(start.get('settled_input')), output_tokens=num(start.get('settled_output')),
        ledger_rmb=num(ledger.get('total_charged_rmb')), node_ms=node).items() if v is not None}


def local_run(root, run_id):
    out = empty(run_id, run_id.replace('-', ' ').upper(), 'controller')
    registration = read_json(safe(root, 'registration.json'))
    # A reviewed holdout run is deliberately never projected, even by explicit path.
    settlement = read_json(safe(root, 'settlement.json')) if registration else {}
    if registration and (registration.get('full_acceptance') is not False or
                         registration.get('holdout_review_sha256') is not None or
                         registration.get('scope') != 'one_engineering_exploration_without_holdout' or
                         settlement.get('holdout_released') is True):
        out['status'] = 'scope_not_supported'
        out['notices'].append('Only engineering runs with no holdout are eligible.')
        return out
    if not registration:
        out['notices'].append('Waiting for an eligible run registration.')
        return out
    out['status'] = 'no_terminal_receipt'
    out['source_ref'] = pin(registration.get('approval_sha256'))
    limit = mapping(registration.get('limits'))
    out['limits'] = {k: v for k in ('input', 'output', 'per_turn_output', 'turns', 'wall_ms') if (v := num(limit.get(k))) is not None}
    rows, objects, clipped = journal(safe(root, 'run', 'trajectory.sqlite3'))
    if clipped:
        out['notices'].append(f'Projection capped at the first {MAX_EVENTS:,} journal rows.')
    seen, measured, comparison, pending = {}, {}, {}, None
    session, node, node_known, order, reserved = None, 0, True, 0, 0
    unknown = 0
    for ordinal, key, row in rows:
        identity = f'journal/{ordinal}'
        op = row.get('operation')
        engine, pair = pin(row.get('engine')), num(row.get('measurement_pair'))
        e = None
        if op == 'engine_registered' and engine:
            iteration = engine_version(out, seen, engine, pin(row.get('parent')))
            e = event(identity, 'engine_registered', engine=engine, parent=pin(row.get('parent')), iteration=iteration)
        elif op == 'worker_reserved' and allowed(row.get('worker_operation'), WORKER_OPS) and engine:
            order += 1
            start, _ = worker_clocks(root, order)
            pending = (order, engine, row['worker_operation'], num(row.get('reserved_ms')), pair)
            reserved = pending[3] or 0
            start = bound_start(start, pending)
            e = event(identity, 'worker_reserved', engine=engine, operation=pending[2], pair=pair,
                      status='reserved', duration_ms=pending[3], wall_ms=num(start.get('elapsed_ms')))
        elif op == 'worker_settled':
            receipt = pin(row.get('evidence'))
            result = objects.get(pin(row.get('result_object')), {})
            typ, state, eng = result.get('operation'), status(result.get('status')), pin(result.get('engine'))
            # An independently linked object can be shown; resource time additionally requires a matching dispatch.
            if receipt and receipt == pin(result.get('evidence')) and allowed(typ, WORKER_OPS) and eng:
                runtime = num(result.get('node_ms'))
                if runtime is not None:
                    node += runtime
                else:
                    node_known = False
                data = mapping(result.get('data'))
                start, end = worker_clocks(root, pending[0], receipt) if pending and pending[1:3] == (eng, typ) else ({}, {})
                if pending:
                    start = bound_start(start, pending)
                if not start or pin(end.get('result_sha256')) != sha_bytes(canonical(result).encode('utf-8')):
                    end = {}
                info = clock(start, end, node if node_known else None)
                q = {k: v for k, v in mapping(data.get('case_outcomes')).items() if k in CASES and status(v)}
                e = event(identity, 'worker_settled', 'verified', engine=eng, operation=typ, status=state,
                          duration_ms=runtime, evidence=receipt, build=pin(result.get('build')),
                          pair=num(data.get('measurement_pair')), wall_ms=info.get('wall_ms'),
                          failure=state in FAILURES, cases=q or None)
                if typ == 'develop_measure' and state == 'observed':
                    measured[receipt] = (e, metrics(data.get('scenes')), info)
            else:
                unknown += 1
                node_known = False
            pending, reserved = None, 0
        elif op == 'worker_uncertain':
            operation = allowed(row.get('worker_operation'), WORKER_OPS)
            start, end = worker_clocks(root, pending[0]) if pending and pending[1:3] == (engine, operation) else ({}, {})
            elapsed = num(end.get('elapsed_ms')) if pending and bound_start(start, pending) and end.get('settlement') == 'uncertain' else None
            e = event(identity, 'worker_uncertain', engine=engine, status='uncertain', failure=True,
                      wall_ms=elapsed, operation=operation)
            if pending and pending[3] is not None:
                out['resources']['unresolved_node_ms'] = out['resources'].get('unresolved_node_ms', 0) + pending[3]
                node_known = False
            pending, reserved = None, 0
        elif op == 'development_reference':
            ref = pin(row.get('evidence'))
            if ref in measured and measured[ref][0]['engine'] == engine:
                out['baseline'] = engine
                e = event(identity, 'reference', 'verified', engine=engine, evidence=ref)
        elif op == 'development_comparison':
            ref, receipt = pin(row.get('reference')), pin(row.get('receipt'))
            if receipt and ref and receipt not in comparison and receipt in measured and ref in measured:
                completed, scenes, info = measured[receipt]
                base, baseline_scenes, _ = measured[ref]
                if engine and completed['engine'] == engine and out['baseline'] == base['engine']:
                    aggregate = num(row.get('weighted_score'))
                    method = 'weighted_score' if aggregate is not None else 'equal_scene_geomean'
                    if aggregate is None:
                        aggregate = num(row.get('equal_scene_geomean'))
                    point = dict(id=receipt, engine=engine, reference=ref, iteration=engine_version(out, seen, engine),
                                 index=len(out['points'])+1, session=session, build=completed.get('build'),
                                 pair=completed.get('pair'), aggregate=aggregate, aggregate_kind=method,
                                 ratios=ratios(row.get('scene_ratios')), scenes=scenes, reference_scenes=baseline_scenes,
                                 resources=info, provisional=row.get('gain_status')=='provisional_pending_weight_review', known_quality=True)
                    out['points'].append(point)
                    comparison[receipt] = point
                    e = event(identity, 'comparison', 'verified', engine=engine, evidence=receipt, point_id=receipt,
                              point_index=point['index'], wall_ms=info.get('wall_ms'), status='observed')
        elif op in ('development_timing_input_mismatch', 'development_workload_drift', 'development_timed_output_mismatch', 'development_output_replay_mismatch'):
            evidence = pin(row.get('candidate'))
            e = event(identity, op, 'verified', engine=engine, evidence=evidence, status='violation', failure=True,
                      wall_ms=measured[evidence][2].get('wall_ms') if evidence in measured else None)
        elif op in ('checkpoint_saved', 'checkpoint_imported', 'checkpoint_import_refused'):
            e = event(identity, op, engine=engine, evidence=pin(row.get('checkpoint')), failure=op.endswith('refused'))
        elif op in ('research_search', 'research_fetch', 'research_read'):
            e = event(identity, 'research', operation=op, evidence=pin(row.get('response_sha256')),
                      request=pin(row.get('request_sha256')), bytes=num(row.get('response_bytes')), cost_rmb=num(row.get('charge_rmb')))
        elif op == 'harness_submit':
            if h := pin(row.get('version')):
                e = event(identity, op, 'agent', harness=h)
        elif op == 'harness_activate':
            if (h := pin(row.get('new'))) and (old := pin(row.get('old'))):
                out['active_harness'] = h
                e = event(identity, op, 'agent', harness=h, previous_harness=old, status='active')
        elif op in ('segment_started', 'segment_settled', 'launcher_failed', 'operator_interrupted', 'exploration_paused'):
            if num(row.get('segment')) is not None:
                session = num(row['segment'])
            h = pin(row.get('harness'))
            if h and op == 'segment_started':
                out['active_harness'] = h
            e = event(identity, op, engine=engine, harness=h, evidence=pin(row.get('checkpoint')),
                      cause=status(row.get('cause')), failure=op in ('launcher_failed', 'operator_interrupted'))
        elif op == 'tool_refused':
            tool = row.get('tool')
            e = event(identity, 'tool_refused', operation=allowed(tool, TOOLS), status='rejected', failure=True)
        elif isinstance(key, str) and (match := re.fullmatch(r'level3/segment/(\d{1,6})/(start|end)', key)):
            session = int(match[1])
            continuation = allowed(row.get('kind'), CONTINUATIONS)
            cause = status(row.get('cause'))
            e = event(identity, f'session_{match[2]}', evidence=pin(row.get('checkpoint')), cause=cause,
                      continuation=continuation, next_session=num(row.get('next_segment')),
                      failure=match[2] == 'end' and cause in FAILURES)
        elif key in ('level3/delivery', 'level3/weight_review_nomination'):
            chosen = mapping(row.get('chosen'))
            final_engine = pin(chosen.get('engine'))
            state = 'delivered' if key.endswith('/delivery') else 'pending_review'
            h = pin(row.get('active_harness'))
            if h:
                out['active_harness'] = h
            if final_engine:
                quality = pin(chosen.get('quality'))
                # Delivery names an engine, build and quality evidence, not a timing reading.
                out['delivery'] = dict(engine=final_engine, evidence=quality, disposition=state, point_id=None)
                e = event(identity, 'delivery', 'agent', engine=final_engine, build=pin(chosen.get('build')),
                          evidence=quality, harness=h, status=state)
        if e:
            e.update(sequence=ordinal)
            if session is not None:
                e['session'] = session
            if e.get('engine') in seen:
                e['iteration'] = seen[e['engine']]
            out['events'].append(e)
    if unknown:
        out['notices'].append(f'{unknown} unbound Worker settlements omitted.')
    if not rows:
        out['notices'].append('No controller Journal events yet.')
    out['resources'].update(settled_node_ms=node if order else None, reserved_node_ms=reserved if order else None,
                            node_ms=node if order and node_known and not reserved else None)
    model_rows = collect_model(out, root)
    collect_terminal(out, root, settlement)
    collect_operator_measurements(out, root, seen)
    annotate_elapsed(out, root)
    if not model_rows:
        out['notices'].append('No projected model receipts yet.')
    return out


def collect_model(out, root):
    totals = {k: 0 for k in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cache_hit_tokens', 'cache_miss_tokens')}
    coverage = {k: 0 for k in totals}
    count = settled = unresolved = start_only = 0
    # Numeric filenames and a single directory level form the collection allowlist.
    dirs = sorted((p for p in root.glob('model-segment-*') if re.fullmatch(r'model-segment-\d{4,6}', p.name)),
                  key=lambda p: int(p.name.split('-')[-1]))
    for d in dirs:
        d = safe(root, d.name)
        session = int(d.name.split('-')[-1])
        for n in sorted({int(m[1]) for p in d.glob('*.json') if (m := re.fullmatch(r'(\d{4,8})-(?:result|start)\.json', p.name))}):
            if count >= MAX_MODEL_ROWS:
                out['notices'].append(f'Model receipt projection capped at {MAX_MODEL_ROWS:,} rows.')
                break
            start = read_json(safe(root, d.name, f'{n:04}-start.json'))
            result = read_json(safe(root, d.name, f'{n:04}-result.json'))
            if not (start or result):
                continue
            count += 1
            u = mapping(result.get('usage'))
            cache = mapping(u.get('reported_cache_usage'))
            stamp = ts(result.get('started_utc')) or ts(start.get('started_utc'))
            actual = {k: num(v) for k, v in dict(input_tokens=u.get('input_tokens'), output_tokens=u.get('output_including_reasoning'),
                      reasoning_tokens=u.get('reasoning_tokens'), cache_hit_tokens=cache.get('hit'), cache_miss_tokens=cache.get('miss')).items()}
            outcome = (status(u.get('result')) or status(result.get('rejection')) or
                       ('started' if not result else 'settled'))
            settlement = status(u.get('usage'))
            if settlement == 'settled':
                settled += 1
                for k, v in actual.items():
                    if v is not None:
                        totals[k] += v
                        coverage[k] += 1
            elif not result:
                start_only += 1
            elif not result.get('rejection'):
                unresolved += 1
            reject = mapping(result.get('rejection'))
            outcome = status(reject.get('reason')) or status(reject.get('cause')) or (outcome if not reject else 'request_rejected')
            finish = status(u.get('finish'))
            generation = mapping(u.get('generation_timing'))
            out['events'].append(event(f'model/{session}/{n}', 'model_request', session=session, attempt=n, status=outcome,
                 at_ms=stamp, duration_ms=num(u.get('wall_ms')), finish=finish, usage_status=settlement,
                 failure=outcome in FAILURES or finish == 'length' or settlement == 'usage_uncertain',
                 first_delta_ms=num(generation.get('first_delta_ms')), last_delta_ms=num(generation.get('last_delta_ms')),
                 input_reserved=num(u.get('input_reserved')), output_reserved=num(u.get('output_reserved')),
                 ledger_rmb=num(mapping(u.get('api_cost')).get('total_charged_rmb')), **actual))
        if count >= MAX_MODEL_ROWS:
            break
    if count:
        out['resources'].update(requests=settled, model_records=count, unresolved_requests=unresolved, requests_without_result=start_only,
                                usage_coverage=coverage,
                                **{k: totals[k] if coverage[k] else None for k in totals})
    return count


def collect_terminal(out, root, settlement):
    if settlement:
        out['status'] = status(settlement.get('status')) or 'settlement_recorded'
        usage = mapping(settlement.get('model_usage'))
        settled = mapping(usage.get('settled'))
        costs = mapping(settled.get('fee_estimate_rmb'))
        node = mapping(settlement.get('worker_cost'))
        ledger = mapping(settlement.get('api_cost'))
        resources = dict(wall_ms=settlement.get('elapsed_ms'), node_ms=node.get('charged_node_ms', settlement.get('node_ms')),
                         settled_node_ms=node.get('settled_node_ms'), reserved_node_ms=node.get('unsettled_node_reservation_ms'),
                         ledger_rmb=ledger.get('total_charged_rmb'), ledger_limit_rmb=ledger.get('limit_rmb'),
                         ledger_outstanding_rmb=ledger.get('outstanding_reservation_rmb'), input_tokens=settled.get('input_tokens'),
                         output_tokens=settled.get('output_tokens'), reasoning_tokens=settled.get('reported_reasoning_tokens'),
                         cache_hit_tokens=settled.get('reported_cache_hit_tokens'), cache_miss_tokens=settled.get('reported_cache_miss_tokens'),
                         requests=settled.get('requests'), unresolved_requests=mapping(usage.get('unresolved')).get('requests'),
                         fee_idle=mapping(costs.get('idle')).get('lower'), fee_peak=mapping(costs.get('peak')).get('upper'))
        out['resources'].update({k: v for k, raw in resources.items() if (v := num(raw)) is not None}, invoice_verified=False)
    terminal = read_json(safe(root, 'run', 'result.json'), cap=16 << 20)
    out['stop_cause'] = status(terminal.get('stop_cause'))
    if out['status'] == 'incomplete' and out['stop_cause'] == 'operator_pause_after_segment':
        out['status'] = 'paused'
    if settlement:
        out['events'].append(event('run/settlement', 'run_settlement', status=out['status'], cause=out['stop_cause'],
                                   wall_ms=num(settlement.get('elapsed_ms')),
                                   failure=(out['stop_cause'] in FAILURES or out['status'] in FAILURES or out['status'] == 'incomplete')))
    for segment in terminal.get('segments', []) if isinstance(terminal.get('segments'), list) else []:
        if not isinstance(segment, dict) or (session := num(segment.get('segment'))) is None:
            continue
        report = mapping(segment.get('report'))
        h = pin(report.get('snapshot'))
        if h:
            out['events'].append(event(f'pi/{session}/attachment', 'session_harness_attached', session=session, harness=h))
        pi_rows = report.get('pi_tool_events')
        if not isinstance(pi_rows, list):
            continue
        if len(pi_rows) > 2000:
            out['notices'].append(f'Pi tool spans for session {session} capped at 2,000 boundaries.')
        opened = {}
        for index, record in enumerate(pi_rows[:2000]):
            record = mapping(record)
            tool, name, hashed = record.get('tool'), record.get('event'), pin(record.get('call_sha256'))
            if (not allowed(tool, TOOLS) or not hashed or (stamp := num(record.get('utc_ns'))) is None
                    or stamp / 1e6 > MAX_EPOCH_MS):
                continue
            if name == 'tool_execution_start':
                opened[hashed] = stamp
            elif name == 'tool_execution_end' and (start := opened.pop(hashed, None)) is not None and stamp >= start:
                out['events'].append(event(f'pi/{session}/{index}', 'pi_tool', session=session, operation=tool, at_ms=start/1e6,
                                           duration_ms=(stamp-start)/1e6, evidence=hashed))



def collect_operator_measurements(out, root, seen):
    """Project the host operator's optional sidecar, bound to this approval.

    The sidecar lives next to the run root and never inherits its resource clock.
    """
    sidecar = read_json(safe(root.parent, root.name + '.observatory-operator.json'), cap=1 << 20)
    records = sidecar.get('measurements')
    if (sidecar.get('schema') != 'ee-observatory-operator-v1' or sidecar.get('run') != out['id']
            or not out['source_ref'] or sidecar.get('approval_sha256') != out['source_ref']
            or not out['baseline'] or not isinstance(records, list) or len(records) > 100):
        return
    reg_doc = read_json(safe(root, 'registration.json'))
    scoring_cfg = mapping(reg_doc.get('business_scoring'))
    run_weights = scoring_cfg.get('weight_percent') or weights_for(scoring_cfg.get('version'))
    published = {point['id'] for point in out['points']}
    for row in map(mapping, records):
        engine, receipt, build, quality, ref = (pin(row.get(key)) for key in ('engine', 'timing', 'build', 'quality', 'reference'))
        case_results = mapping(row.get('cases'))
        if (not all((engine, receipt, build, quality, ref)) or receipt in published or receipt == ref
                or engine not in seen or engine == out['baseline'] or row.get('reference_engine') != out['baseline']
                or row.get('evidence_origin') != 'operator_replay' or row.get('verdict') != 'measured'
                or row.get('weight_audit') != 'pending_if_gain'
                or not REPLAY_CASES <= case_results.keys() or any(v != 'pass' for v in case_results.values())):
            continue
        scene_ratios = ratios(row.get('ratios'))
        reported = num(row.get('weighted_score'))
        if set(scene_ratios) != set(SCENES) or min(scene_ratios.values()) <= 0 or reported is None:
            continue
        score = weighted_score(scene_ratios, run_weights)
        if not isclose(score, reported, rel_tol=1e-9):
            continue
        index = len(out['points']) + 1
        out['points'].append(dict(id=receipt, engine=engine, reference=ref, iteration=seen[engine], index=index,
                                  build=build, aggregate=score, aggregate_kind='weighted_score', ratios=scene_ratios,
                                  scenes={}, reference_scenes={}, provisional=True, known_quality=True, quality=quality,
                                  origin='operator_replay'))
        out['events'].append(event(f'operator/{receipt}', 'operator_measurement', 'verified', engine=engine,
                                   evidence=receipt, point_id=receipt, point_index=index, iteration=seen[engine],
                                   build=build, status='observed'))
        published.add(receipt)


TIMELINE_KINDS = frozenset(('pause', 'resume', 'infrastructure_change', 'infrastructure_fault'))
TIMELINE_REASONS = frozenset(('infrastructure_repair', 'worker_fault', 'operator_pause', 'unplanned_outage',
                              'generation_truncated', 'transport_failure', 'network_failure'))


def annotate_elapsed(out, root):
    """Optional trusted operator ledger, outside the controller's output directory.

    Ordinary checkpoint handoffs and unstamped incidents are never assumed to be
    pauses. Missing/invalid ledgers leave the effective-time axis unavailable.
    """
    path = safe(root.parent, root.name + '.observatory-timing.json')
    source = read_json(path, cap=1 << 20)
    rows = source.get('events')
    if (source.get('schema') != 'ee-observatory-timing-v1' or not isinstance(rows, list)
            or len(rows) > 4096 or not out['source_ref'] or source.get('approval_sha256') != out['source_ref']):
        return
    start, previous = None, -1
    ranges, projected = [], []
    for index, raw in enumerate(rows, 1):
        row = mapping(raw)
        kind, wall, reason = allowed(row.get('kind'), TIMELINE_KINDS), num(row.get('wall_ms')), allowed(row.get('cause'), TIMELINE_REASONS)
        infra = pin(row.get('infrastructure_sha256'))
        if kind is None or type(wall) is not int or wall < previous or (kind == 'infrastructure_change' and infra is None):
            return
        if kind == 'pause':
            if start is not None or reason is None:
                return
            start = wall
        elif kind == 'resume':
            if start is None:
                return
            ranges.append((start, wall))
            start = None
        previous = wall
        projected.append(event(f'annotations/{index}', kind, wall_ms=wall, cause=reason,
                               evidence=infra, failure=kind == 'infrastructure_fault'))
    if start is not None:
        ranges.append((start, None))

    def adjusted(wall):
        return wall - sum(max(0, (min(wall, end) if end is not None else wall)-begin) for begin, end in ranges)

    out['events'].extend(projected)
    for reading in out['points'] + out['events']:
        values = reading.get('resources', reading)
        if (wall := values.get('wall_ms')) is not None:
            values['effective_ms'] = adjusted(wall)
    if (wall := out['resources'].get('wall_ms')) is not None:
        out['resources']['effective_ms'] = adjusted(wall)
    out['notices'].append('Effective time excludes only explicitly recorded pause intervals; unpaused faults remain included.')
