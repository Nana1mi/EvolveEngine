"""Versioned business scoring for the interactive GB10 free exploration."""
from math import exp, isfinite, log, sqrt
from statistics import mean, stdev

VERSION = 'interactive-tpot-v2'
NATURAL_VERSION = 'interactive-tpot-natural-eos-v3'
CHAT_VERSION = 'interactive-tpot-thinking-chat-v4'
WORKBUDDY_VERSION = 'workbuddy-agent-serving-v1'
WORKBUDDY_V2_VERSION = 'workbuddy-agent-serving-v2'
WORKBUDDY_VERSIONS = {WORKBUDDY_VERSION, WORKBUDDY_V2_VERSION}
CHAT_VERSIONS = {CHAT_VERSION, WORKBUDDY_VERSION, WORKBUDDY_V2_VERSION}
NATURAL_VERSIONS = {NATURAL_VERSION, CHAT_VERSION, WORKBUDDY_VERSION, WORKBUDDY_V2_VERSION}
WEIGHTS = {'W1': 20, 'W2': 70, 'W3': 5, 'W4': 5}
WORKBUDDY_WEIGHTS = {'W1': 15, 'W2': 20, 'W3': 30, 'W4': 35}
WORKBUDDY_V2_WEIGHTS = {'W1': 10, 'W2': 15, 'W3': 25, 'W4_hot': 15, 'W4_cold': 15, 'W5': 20}
WORKBUDDY_V2_REPLAY_SCENES = ('W1', 'W2', 'W3', 'W4', 'W5')
WORKBUDDY_V2_PRIMARY_SCENES = ('W3', 'W4_hot', 'W4_cold', 'W5')
GUARDRAIL_KEYS = ('w4_cold_e2e', 'g64k_ttft', 'g64k_decode', 'p95_itl')
P95_ITL_RELAXED_FLOOR = 0.80
SLA_TARGETS_TPS = (40.0, 50.0)
MIN_PAIRS = 6
PRECISION_TARGET = 1.015
WORKBUDDY_V2_PRECISION_CAP = 1.025
# Six alternating E0/E0 pairs on GB10, Graph defaults and the real-text workload.
CALIBRATION = '7f2dbdba130c6b0815e6b7679c55cf69ba0bfa59de23727c428345e761bc3833'
LOG_ENVELOPE = dict(W1=0.02072928850556721, W2=0.01, W3=0.010536495536740298, W4=0.016593829860838127)
FLOORS = {scene: max(0.97, exp(-width)) for scene, width in LOG_ENVELOPE.items()}
T95_DF5 = 2.570581835636305  # conservative for subsequent balanced batches of >6 pairs
LENGTH_DRIFT = 0.10
STOP_SHARE_DRIFT = 0.10


def weights_for(version):
    if version == WORKBUDDY_V2_VERSION:
        return WORKBUDDY_V2_WEIGHTS
    return WORKBUDDY_WEIGHTS if version == WORKBUDDY_VERSION else WEIGHTS


def workbuddy_v2_precision_target(logs):
    """Preregistered rule: min(2.5%, max(1.5%, 1.2 * observed 6-pair 95% CI half-width))."""
    if len(logs) < MIN_PAIRS or not all(type(v) in (int, float) and isfinite(v) for v in logs):
        raise ValueError('expected at least 6 finite calibration log-ratios')
    half = T95_DF5 * stdev(logs) / sqrt(len(logs))
    return min(WORKBUDDY_V2_PRECISION_CAP, max(PRECISION_TARGET, exp(1.2 * half)))


def weighted_score(ratios, weights=None):
    table = WEIGHTS if weights is None else weights
    if set(ratios) != set(table) or not all(type(v) in (int, float) and isfinite(v) and v > 0 for v in ratios.values()):
        raise ValueError('expected four positive finite scene ratios')
    total = sum(table.values())
    if total <= 0:
        raise ValueError('invalid scene weight total')
    return exp(sum(weight * log(ratios[scene]) for scene, weight in table.items()) / total)


def workbuddy_sla_summary(scenes, guardrails=None):
    """Separate business SLA attainment telemetry from positive-ratio exploration scoring."""
    summary = {}
    for scene in WORKBUDDY_V2_REPLAY_SCENES:
        row = scenes.get(scene)
        if not isinstance(row, dict):
            continue
        entry = {}
        for key in ('share_ge_40_tps', 'share_ge_50_tps', 'ttft_ms', 'p95_itl_ms', 'tpot_ms',
                    'output_toks_per_s', 'unfinished_count', 'cached_tokens_min', 'cached_tokens_mean',
                    'w4_hot_decode_tps', 'w4_cold_ttft_ms', 'w4_cold_e2e_ms', 'hot_cached_tokens_min'):
            val = row.get(key)
            if type(val) in (int, float) and isfinite(val):
                entry[key] = val
        if entry:
            summary[scene] = entry
    if isinstance(guardrails, dict) and isinstance(guardrails.get('G64K'), dict):
        g = guardrails['G64K']
        g_entry = {k: g[k] for k in ('ttft_ms', 'decode_tokens_per_s', 'tpot_ms', 'e2e_ms', 'unfinished_count', 'cached_tokens_min')
                   if type(g.get(k)) in (int, float) and isfinite(g[k])}
        if g_entry:
            summary['G64K'] = g_entry
    return summary


def paired_w3_ttft(a, b):
    if not a or not b or set(a)!=set(b):
        raise ValueError('W3 needs the same frozen requests on both arms')
    details = {}
    for key in a:
        left, right = a[key], b[key]
        x, y = left['raw_ttft_ms'], right['raw_ttft_ms']
        if left['input_tokens']!=right['input_tokens'] or not all(isfinite(v) and v>0 for v in (x,y)):
            raise ValueError('incomparable W3 inputs or TTFT')
        details[key] = dict(input_tokens=left['input_tokens'], e0_raw_ttft_ms=x, candidate_raw_ttft_ms=y, ratio=x/y,
                            e0_visible_answer_ttft_ms=left['visible_answer_ttft_ms'],
                            candidate_visible_answer_ttft_ms=right['visible_answer_ttft_ms'])
    return exp(mean(log(row['ratio']) for row in details.values())),details


def policy():
    return dict(version=VERSION, weight_percent=dict(WEIGHTS), scene_floors=dict(FLOORS),
                e0_calibration_sha256=CALIBRATION, minimum_pairs=MIN_PAIRS,
                precision_target_ratio=PRECISION_TARGET, ordering='alternating_baseline_first_candidate_first')


# Descriptive audit trigger, not an acceptance limit or confidence interval.
REPLAY_AUDIT_MARGIN = .10


def replay_agreement(left, right):
    if set(left) != set(right) or not left: raise ValueError('unmatched replay scenes')
    out = {}
    for scene, rows in left.items():
        if not rows or rows.keys() != right[scene].keys(): raise ValueError('unmatched replay requests')
        pairs = [(a, right[scene][k]) for k,a in rows.items()]
        total = sum(max(len(a),len(b)) for a,b in pairs)
        if any(not a or not b for a,b in pairs): raise ValueError('empty replay')
        same = sum(sum(x==y for x,y in zip(a,b)) for a,b in pairs)
        out[scene] = dict(requests=len(pairs),exact=sum(a==b for a,b in pairs),same_positions=same,
                          positions=total,rate=same/total)
    return out


def replay_audit_floors(pairs):
    if len(pairs) != MIN_PAIRS: raise ValueError('incomplete E0 agreement calibration')
    first_obs = pairs[0][0]['paired_replay_agreement']
    expected_scenes = WORKBUDDY_V2_REPLAY_SCENES if set(first_obs) == set(WORKBUDDY_V2_REPLAY_SCENES) else tuple(WEIGHTS)
    rates = {k:[] for k in expected_scenes}
    for a,b in pairs:
        obs = a['paired_replay_agreement']
        if obs != b['paired_replay_agreement'] or obs.keys()!=rates.keys(): raise ValueError('unmatched E0 replay coverage')
        for k,row in obs.items():
            if (any(type(row.get(x)) is not int for x in ('same_positions','positions','requests','exact')) or
                not 0<=row['exact']<=row['requests']==len(a['timed_request_outputs'][k]) or
                not 0<row['requests']<=row['positions'] or not 0<=row['same_positions']<=row['positions'] or
                row['rate']!=row['same_positions']/row['positions']): raise ValueError('invalid E0 replay counts')
            rates[k].append(row['rate'])
    return {k:max(0., min(v)-REPLAY_AUDIT_MARGIN) for k,v in rates.items()}


def natural_policy(calibration_sha256, workload_sha256, envelope, *, chat_corpus=None, replay_floors=None,
                   version=None, scene_precision_targets=None, guardrail_envelope=None):
    chosen = version or (CHAT_VERSION if chat_corpus else NATURAL_VERSION)
    if chosen not in NATURAL_VERSIONS:
        raise ValueError('unknown natural-EOS scoring version')
    w_table = weights_for(chosen)
    max_env = 0.08 if chosen == WORKBUDDY_V2_VERSION else 0.05
    if set(envelope) != set(w_table) or not all(type(v) in (int, float) and 0 < v <= max_env for v in envelope.values()):
        raise ValueError('invalid natural-EOS calibration envelope')
    if chosen in CHAT_VERSIONS and not chat_corpus:
        raise ValueError('chat scoring policy requires chat corpus')
    result = dict(policy(), version=chosen, weight_percent=dict(w_table), timing_mode='natural_eos',
                  timing_workload_sha256=workload_sha256, e0_calibration_sha256=calibration_sha256,
                  scene_log_envelope=dict(envelope), scene_floors={k: exp(-v) for k, v in envelope.items()})
    replay_keys = set(WORKBUDDY_V2_REPLAY_SCENES) if chosen == WORKBUDDY_V2_VERSION else set(WEIGHTS)
    if replay_floors is not None:
        if not chat_corpus or set(replay_floors) != replay_keys or not all(type(x) in (float,int) and 0<=x<=1 for x in replay_floors.values()):
            raise ValueError("invalid replay audit floors")
        result["replay_audit_floors"]=dict(replay_floors)
    if chat_corpus:
        result.update(chat_corpus_sha256=chat_corpus, enable_thinking=True)
    if chosen in WORKBUDDY_VERSIONS:
        result['sla_targets_tps'] = list(SLA_TARGETS_TPS)
    if chosen == WORKBUDDY_V2_VERSION:
        if (type(scene_precision_targets) is not dict or set(scene_precision_targets) != set(WORKBUDDY_V2_PRIMARY_SCENES)
                or not all(type(v) in (int, float) and isfinite(v) and PRECISION_TARGET <= v <= WORKBUDDY_V2_PRECISION_CAP
                           for v in scene_precision_targets.values())):
            raise ValueError('workbuddy-agent-serving-v2 requires calibrated scene_precision_targets in [1.015, 1.025]')
        if (type(guardrail_envelope) is not dict or set(guardrail_envelope) != set(GUARDRAIL_KEYS)
                or not all(type(v) in (int, float) and 0 < v <= 0.25 for v in guardrail_envelope.values())):
            raise ValueError('workbuddy-agent-serving-v2 requires calibrated guardrail_envelope')
        result['scene_precision_targets'] = {k: float(v) for k, v in scene_precision_targets.items()}
        result['guardrail_log_envelope'] = {k: float(v) for k, v in guardrail_envelope.items()}
        result['guardrail_floors'] = {
            k: (min(P95_ITL_RELAXED_FLOOR, exp(-v)) if k == 'p95_itl' else exp(-v))
            for k, v in guardrail_envelope.items()
        }
    return result


def resolve_policy(scoring):
    if scoring == VERSION:
        return policy()
    if isinstance(scoring, dict) and scoring.get('version') in NATURAL_VERSIONS:
        expected = natural_policy(scoring['e0_calibration_sha256'], scoring['timing_workload_sha256'],
                                  scoring['scene_log_envelope'], chat_corpus=scoring.get('chat_corpus_sha256'),
                                  replay_floors=scoring.get('replay_audit_floors'), version=scoring['version'],
                                  scene_precision_targets=scoring.get('scene_precision_targets'),
                                  guardrail_envelope=scoring.get('guardrail_log_envelope'))
        if expected == scoring:
            return expected
    raise ValueError('unknown scoring policy')


def compare_output_work(reference, candidate):
    """Compare matched requests after independent same-Engine replay established authenticity."""
    summaries = {}
    ref_scenes = reference['timed_request_outputs']
    expected_scenes = WORKBUDDY_V2_REPLAY_SCENES if set(ref_scenes) == set(WORKBUDDY_V2_REPLAY_SCENES) else tuple(WEIGHTS)
    for scene in expected_scenes:
        a, b = (arm['timed_request_outputs'][scene] for arm in (reference, candidate))
        if not a or a.keys() != b.keys():
            raise ValueError('output evidence does not cover the same timed requests')
        matched = [(a[key], b[key]) for key in a]
        count = len(matched)
        lengths = [sum(pair[arm]['tokens'] for pair in matched) for arm in (0,1)]
        stopped = [sum(pair[arm]['finish_reason'] == 'stop' for pair in matched) for arm in (0,1)]
        early = [k for k in a if a[k]['finish_reason'] == 'length' and b[k]['finish_reason'] == 'stop'
                 and b[k]['tokens'] < a[k]['tokens']]
        mismatched = sum(x['sha256'] != y['sha256'] for x,y in matched)
        ratio = lengths[1] / lengths[0]
        stop_difference = abs(stopped[1] - stopped[0]) / count
        screened = scene != 'W3' and reference.get('chat_corpus_sha256')
        stop_limit = STOP_SHARE_DRIFT if screened else max(STOP_SHARE_DRIFT, 1/count)
        summaries[scene] = dict(requests=count, mismatched=mismatched, mismatch_rate=mismatched/count,
                                e0_output_tokens=lengths[0], candidate_output_tokens=lengths[1],
                                output_length_ratio=ratio, e0_eos=stopped[0], candidate_eos=stopped[1],
                                e0_at_cap_candidate_early_eos=early, early_eos_rate=len(early)/count,
                                eos_share_difference=stop_difference,
                                comparable=abs(ratio-1) <= LENGTH_DRIFT and stop_difference <= stop_limit and
                                           (not screened or len(early)/count <= stop_limit))
    return summaries


def calibration_output_work(first, second, corpus):
    """Recompute paired work from observations, not a saved comparable/pass flag."""
    from .v3_harness import canonical, digest
    arms = {row['arm']:row for row in (first, second)}
    if set(arms) != {'e0a','e0b'}:
        raise ValueError('expected two different E0 calibration arms')
    for arm in arms.values():
        outputs = {}
        scenes = arm['timed_request_outputs']
        expected_keys = set(WORKBUDDY_V2_REPLAY_SCENES) if set(scenes) == set(WORKBUDDY_V2_REPLAY_SCENES) else set(WEIGHTS)
        if scenes.keys() != expected_keys or not all(scenes.values()):
            raise ValueError('missing timed output scenes')
        for requests in scenes.values():
            for key, r in requests.items():
                if (key in outputs or any(type(h) is not str or len(h) != 64 or
                        any(c not in '0123456789abcdef' for c in h) for h in (key,r['sha256'])) or
                        type(r['tokens']) is not int or r['tokens'] <= 0 or r['finish_reason'] not in ('length','stop')):
                    raise ValueError('invalid calibration output')
                outputs[key] = [r['sha256'], r['tokens'], r['finish_reason']]
        if digest(canonical(outputs)) != arm['timed_outputs_sha256']:
            raise ValueError('calibration output fingerprint differs')
    # Alternate measurement order, never the denominator of the asymmetric drift tests.
    return compare_output_work(dict(arms['e0a'], chat_corpus_sha256=corpus), arms['e0b'])


def summarize_build(jobs, engine, build, scoring=VERSION):
    """Use every verified comparison of this build; keep diagnostics at any sample size."""
    registered = resolve_policy(scoring)
    floors = registered['scene_floors']
    weights = registered['weight_percent']
    wb_v2 = registered['version'] == WORKBUDDY_V2_VERSION
    if wb_v2:
        primary_scenes = WORKBUDDY_V2_PRIMARY_SCENES
    elif registered['version'] == WORKBUDDY_VERSION:
        primary_scenes = ('W3', 'W4')
    else:
        primary_scenes = ('W1', 'W2')
    guardrail_floors = registered.get('guardrail_floors') or {}
    precision_targets = registered.get('scene_precision_targets') or {}
    positions = {row['evidence']: index for index, row in enumerate(jobs) if 'evidence' in row}
    pairs, guardrail_pairs, ids, inputs = [], [], set(), set()
    mismatch = False
    workload_drift = False
    unfinished_violation = False
    first_count = 0
    for index, row in enumerate(jobs):
        if (row.get('engine'), row.get('build')) != (engine, build):
            continue
        mismatch |= row.get('scoring_status') in {'timed_output_mismatch','output_replay_mismatch'}
        workload_drift |= row.get('scoring_status') == 'workload_drift'
        point = row.get('performance')
        if point is None:
            continue
        pair, fingerprint = row['data']['measurement_pair'], row['data']['measurement_inputs_sha256']
        if (pair in ids or (registered['version'] not in CHAT_VERSIONS and fingerprint in inputs)
                or point['reference'] not in positions):
            raise ValueError('duplicate or unpaired business measurement')
        ids.add(pair)
        inputs.add(fingerprint)
        first_count += index < positions[point['reference']]
        ratios = point['scene_ratios']
        weighted_score(ratios, weights)
        pairs.append(ratios)
        if wb_v2:
            g_ratios = point.get('guardrail_ratios')
            if (type(g_ratios) is not dict or set(g_ratios) != set(GUARDRAIL_KEYS)
                    or not all(type(v) in (int, float) and isfinite(v) and v > 0 for v in g_ratios.values())):
                raise ValueError('workbuddy-agent-serving-v2 requires positive finite guardrail ratios')
            guardrail_pairs.append(g_ratios)
            if point.get('unfinished_count', 0) != 0:
                unfinished_violation = True
    count = len(pairs)
    initial_status = 'timed_output_mismatch' if mismatch else 'workload_drift' if workload_drift else 'needs_repeats'
    result = dict(version=registered['version'], pairs=count, candidate_first=first_count, eligible_for_delivery=False,
                  status=initial_status, scene_ratios={}, intervals={},
                  weighted_score=None, below_floor=[], uncertain_scenes=[])
    if wb_v2:
        result.update(guardrail_ratios={}, guardrail_intervals={})
    if not count:
        return result
    for scene in weights:
        logs = [log(p[scene]) for p in pairs]
        center = mean(logs)
        result['scene_ratios'][scene] = exp(center)
        if count >= MIN_PAIRS:
            half = T95_DF5 * stdev(logs) / sqrt(count)
            lo, hi = exp(center - half), exp(center + half)
            result['intervals'][scene] = [lo, hi]
            target = precision_targets.get(scene, PRECISION_TARGET)
            if hi < floors[scene]:
                result['below_floor'].append(scene)
            elif lo < floors[scene] or (scene in primary_scenes and half > log(target)):
                result['uncertain_scenes'].append(scene)
    if wb_v2:
        if unfinished_violation:
            result['below_floor'].append('unfinished_count')
        for gkey in GUARDRAIL_KEYS:
            logs = [log(p[gkey]) for p in guardrail_pairs]
            center = mean(logs)
            result['guardrail_ratios'][gkey] = exp(center)
            if count >= MIN_PAIRS:
                half = T95_DF5 * stdev(logs) / sqrt(count)
                lo, hi = exp(center - half), exp(center + half)
                result['guardrail_intervals'][gkey] = [lo, hi]
                if hi < guardrail_floors[gkey]:
                    result['below_floor'].append(gkey)
                elif lo < guardrail_floors[gkey]:
                    result['uncertain_scenes'].append(gkey)
    result['weighted_score'] = weighted_score(result['scene_ratios'], weights)
    if mismatch or workload_drift or count < MIN_PAIRS:
        return result
    if first_count * 2 != count:
        result['status'] = 'needs_balanced_order'
    elif result['below_floor']:
        result['status'] = 'regression'
    elif result['uncertain_scenes']:
        result['status'] = 'inconclusive'
    else:
        result.update(eligible_for_delivery=True, status='eligible')
    return result
