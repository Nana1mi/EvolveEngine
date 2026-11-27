"""Long-context (16K/64K) logprob correctness check against an E0/E0-calibrated BF16 anchor."""
import json
from math import exp, isfinite
from pathlib import Path
import re

from .v3_harness import canonical, digest

SCHEMA = 'level3-long-context-logprob-anchor-v1'
CASES_SCHEMA = 'level3-long-context-logprob-cases-v1'
CASES = ('ctx16k', 'ctx64k')
EXPECTED_LENGTHS = {'ctx16k': 16384, 'ctx64k': 64000}
TOP_K = 64
DEFAULT_MAX_TV_UPPER = 0.05
DEFAULT_MAX_GREEDY_REGRET = 0.05
SHA = re.compile(r'[a-f0-9]{64}')


def _need(ok, why):
    if not ok:
        raise ValueError(why)


def _checked_top_logprobs(raw):
    _need(type(raw) is dict and len(raw) >= 1, 'missing top logprobs map')
    out = {}
    for k, v in raw.items():
        _need(type(k) is str and k.isdigit() and 0 <= int(k) < 131072, 'invalid token id in top_logprobs')
        _need(type(v) in (int, float) and isfinite(v) and v <= 1e-4, 'invalid logprob value')
        out[int(k)] = float(v)
    return out


def verify_cases(cases_path, expected_sha256):
    path = Path(cases_path)
    _need(SHA.fullmatch(str(expected_sha256)) and path.is_file(), 'invalid long-context cases path or hash')
    raw = path.read_bytes()
    _need(digest(raw) == expected_sha256, 'long-context cases hash mismatch')
    doc = json.loads(raw)
    _need(raw == canonical(doc) + b'\n' and type(doc) is dict and doc.get('schema') == CASES_SCHEMA
          and set(doc) == {'schema', 'tokenizer_sha256', 'cases'}, 'invalid long-context cases schema')
    cases = doc['cases']
    _need(type(cases) is dict and set(cases) == set(CASES), 'long-context cases must cover ctx16k and ctx64k')
    for name, expected_len in EXPECTED_LENGTHS.items():
        row = cases[name]
        _need(type(row) is dict and set(row) == {'ids', 'prompt_sha256'}, f'invalid long-context case {name}')
        ids = row['ids']
        _need(type(ids) is list and len(ids) == expected_len
              and all(type(t) is int and 0 <= t < 131072 for t in ids)
              and row['prompt_sha256'] == digest(canonical(ids)),
              f'long-context prompt length or hash mismatch for {name}')
    return doc


def verify_anchor(anchor_path, expected_sha256):
    path = Path(anchor_path)
    _need(SHA.fullmatch(str(expected_sha256)) and path.is_file(), 'invalid long-context anchor path or hash')
    raw = path.read_bytes()
    _need(digest(raw) == expected_sha256, 'long-context anchor hash mismatch')
    doc = json.loads(raw)
    _need(raw == canonical(doc) + b'\n' and type(doc) is dict and doc.get('schema') == SCHEMA
          and set(doc) == {'schema', 'engine', 'calibration', 'cases'}, 'invalid long-context anchor schema')
    cal = doc['calibration']
    _need(type(cal) is dict and set(cal) == {'e0e0_max_tv_upper', 'e0e0_max_regret', 'max_tv_upper', 'max_greedy_regret'}
          and all(type(cal[k]) in (int, float) and isfinite(cal[k]) and cal[k] >= 0 for k in cal)
          and 0 < cal['max_tv_upper'] <= 0.10 and 0 < cal['max_greedy_regret'] <= 0.10,
          'invalid long-context anchor calibration bounds')
    cases = doc['cases']
    _need(type(cases) is dict and set(cases) == set(CASES), 'long-context anchor must cover ctx16k and ctx64k')
    for name, expected_len in EXPECTED_LENGTHS.items():
        row = cases[name]
        _need(type(row) is dict and set(row) == {'prompt_tokens', 'prompt_sha256', 'top1_id', 'top1_logprob', 'top_logprobs'}
              and row['prompt_tokens'] == expected_len and SHA.fullmatch(str(row['prompt_sha256']))
              and type(row['top1_id']) is int and 0 <= row['top1_id'] < 131072
              and type(row['top1_logprob']) in (int, float) and isfinite(row['top1_logprob']),
              f'invalid long-context anchor entry {name}')
        _checked_top_logprobs(row['top_logprobs'])
    return doc


def calibrate_bounds(e0a_cases, e0b_cases):
    """Derive calibrated TV and regret limits from two independent E0 deployments (e0a vs e0b)."""
    tvs, regrets = [], []
    for name in CASES:
        res = compare_distributions(e0a_cases[name], e0b_cases[name])
        tvs.append(res['tv_upper'])
        regrets.append(res['regret'] if res['regret'] is not None else 1.0)
    e0_tv = max(tvs)
    e0_reg = max(regrets)
    _need(e0_tv <= DEFAULT_MAX_TV_UPPER and e0_reg <= DEFAULT_MAX_GREEDY_REGRET,
          f'E0/E0 long-context drift exceeds hard cap (tv={e0_tv:.4f}, regret={e0_reg:.4f})')
    return dict(
        e0e0_max_tv_upper=e0_tv,
        e0e0_max_regret=e0_reg,
        max_tv_upper=min(DEFAULT_MAX_TV_UPPER, max(0.02, 2.0 * e0_tv)),
        max_greedy_regret=min(DEFAULT_MAX_GREEDY_REGRET, max(0.02, 2.0 * e0_reg)),
    )


def compare_distributions(ref_row, cand_row, *, max_tv_upper=DEFAULT_MAX_TV_UPPER,
                          max_greedy_regret=DEFAULT_MAX_GREEDY_REGRET):
    """Exact partial-TV upper bound and greedy regret on top-K logprobs."""
    ref_lp = _checked_top_logprobs(ref_row['top_logprobs'])
    cand_lp = _checked_top_logprobs(cand_row['top_logprobs'])
    ref_p = {k: min(1.0, exp(v)) for k, v in ref_lp.items()}
    cand_p = {k: min(1.0, exp(v)) for k, v in cand_lp.items()}
    r_p = max(0.0, 1.0 - sum(ref_p.values()))
    r_q = max(0.0, 1.0 - sum(cand_p.values()))
    union = set(ref_p) | set(cand_p)
    l1_observed = sum(abs(ref_p.get(k, 0.0) - cand_p.get(k, 0.0)) for k in union)
    tv_upper = 0.5 * (l1_observed + r_p + r_q)
    cand_top1 = cand_row['top1_id']
    ref_top1_lp = float(ref_row['top1_logprob'])
    if cand_top1 in ref_lp:
        regret = max(0.0, ref_top1_lp - ref_lp[cand_top1])
    else:
        regret = float('inf')
    passed = tv_upper <= max_tv_upper and regret <= max_greedy_regret
    return dict(
        passed=passed,
        tv_upper=tv_upper,
        regret=regret if isfinite(regret) else None,
        residual_ref=r_p,
        residual_cand=r_q,
        top1_match=(cand_top1 == ref_row['top1_id']),
        prompt_tokens=ref_row['prompt_tokens'],
    )


def grade_long_ctx(rows, anchor_path, expected_sha256):
    anchor = verify_anchor(anchor_path, expected_sha256)
    cal = anchor['calibration']
    by_case = {}
    for row in rows:
        name = row.get('case')
        _need(name in EXPECTED_LENGTHS and name not in by_case, 'unexpected or duplicate long-context logprob case')
        by_case[name] = row
    _need(set(by_case) == set(CASES), 'missing long-context logprob cases')
    outcomes = {}
    for name in CASES:
        ref_row = anchor['cases'][name]
        cand_row = by_case[name]
        _need(cand_row.get('prompt_tokens') == ref_row['prompt_tokens']
              and cand_row.get('prompt_sha256') == ref_row['prompt_sha256']
              and type(cand_row.get('top1_id')) is int and 0 <= cand_row['top1_id'] < 131072
              and type(cand_row.get('top1_logprob')) in (int, float) and isfinite(cand_row['top1_logprob']),
              f'long-context probe identity mismatch for {name}')
        outcomes[name] = compare_distributions(
            ref_row, cand_row,
            max_tv_upper=cal['max_tv_upper'],
            max_greedy_regret=cal['max_greedy_regret'],
        )
    all_ok = all(item['passed'] for item in outcomes.values())
    return dict(
        status='valid' if all_ok else 'violation',
        anchor_sha256=expected_sha256,
        calibration=cal,
        cases=outcomes,
    )
