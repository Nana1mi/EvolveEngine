"""Research screen v0: explicitly not a production correctness or service-level agreement."""
from math import isfinite

COUNT = 32
MAX_TV = 0.05
BULK_TV = 0.02
BULK_EXCEPTIONS = 3
LARGE_MARGIN = 1.0


def score(by_position):
    """Only supply provenance-checked measurements. Missing rows cannot count as passes."""
    if not all(type(i) is int and 0 <= i < COUNT for i in by_position):
        raise ValueError('unknown position ID')
    missing, hard, bulk, flips = [], [], [], []
    for i in range(COUNT):
        m = by_position.get(i)
        if not isinstance(m, dict):
            missing.append(i)
            continue
        tv, margin, flip = (m.get(k) for k in ('tv', 'ref_margin', 'flip'))
        if (type(tv) not in (float, int) or not isfinite(tv) or not 0 <= tv <= 1
                or type(margin) not in (float, int) or not isfinite(margin) or margin < 0
                or type(flip) is not bool):
            missing.append(i)
            continue
        if tv > MAX_TV:
            hard.append(i)
        if tv > BULK_TV:
            bulk.append(i)
        if margin >= LARGE_MARGIN and flip:
            flips.append(i)
    violation = bool(hard or flips or len(bulk) > BULK_EXCEPTIONS)
    status = 'outside_dev_screen' if violation else ('insufficient_evidence' if missing else 'within_dev_screen')
    return dict(status=status, n=COUNT-len(missing), missing=missing, tv_over_005=hard,
                tv_over_002=bulk, large_margin_flips=flips)
