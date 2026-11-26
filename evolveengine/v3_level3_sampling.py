"""Engineering sampling screen against paired live E0 observations."""
from collections import Counter
from math import log, sqrt

VOCAB = 131072
CASES = (0, 9, 18, 27)
SAMPLES = 2048
EVENT_MARGIN = 0.12


def checked_samples(row):
    if (type(row) is not dict or tuple(sorted(row)) != tuple(sorted(str(i) for i in CASES))
            or any(type(samples) is not list or len(samples) != SAMPLES for samples in row.values())):
        raise ValueError('missing registered sampling coverage')
    for samples in row.values():
        for sequence in samples:
            if (type(sequence) is not list or not 1 <= len(sequence) <= 3 or
                    any(type(t) is not int or not 0 <= t < VOCAB for t in sequence)):
                raise ValueError('invalid independently observed sampled sequence')
    return row


VERSION = 'milestone-sampling-two-look-v1'
ALPHA = .025


def compare(reference, candidate, *, alpha=.05):
    """Split E0 data to choose events, then bound probability differences on separate draws."""
    if alpha not in (.05,ALPHA): raise ValueError("unregistered sampling error budget")
    reference, candidate = checked_samples(reference), checked_samples(candidate)
    tests = []
    for case in CASES:
        selector, baseline = reference[str(case)][:512], reference[str(case)][512:]
        trial = candidate[str(case)]
        heads = [token for token, _ in Counter(s[0] for s in selector).most_common(6)]
        transition = next(iter(Counter(tuple(s[:2]) for s in selector if len(s) >= 2).most_common(1)), (None,0))[0]
        events = [(f'first={token}', lambda s, token=token: s[0] == token) for token in heads]
        events.append(('natural_stop_before_cap', lambda s: len(s) < 3))
        if transition is not None:
            events.append((f'prefix={transition}', lambda s: tuple(s[:2]) == transition))
        for name, predicate in events:
            p = sum(predicate(s) for s in baseline) / len(baseline)
            q = sum(predicate(s) for s in trial) / len(trial)
            half = sqrt(0.5 * (1/len(baseline) + 1/len(trial)) * log(2 * len(CASES) * 8 / alpha))
            tests.append(dict(case=case, event=name, e0=p, candidate=q, difference=abs(p-q),
                              lower_bound=max(0.,abs(p-q)-half), upper_bound=abs(p-q)+half))
    return dict(status='pass' if all(t['upper_bound'] <= EVENT_MARGIN for t in tests) else 'inconclusive_or_different',
                margin=EVENT_MARGIN, simultaneous_error=alpha, events=tests,
                scope='empirical_selected_event_equivalence')


def decision(report):
    if report['simultaneous_error']!=ALPHA or not report['events']: raise ValueError('unregistered sampling look')
    if report['status']=='pass': return 'pass'
    if any(r['lower_bound']>EVENT_MARGIN for r in report['events']): return 'different'
    return 'inconclusive'


def challenge(spec):
    if (type(spec) is not dict or spec.keys()!={'milestone','stage','role'} or
        spec['milestone'] not in ('historical_best','final_nomination') or type(spec['stage']) is not int or
        spec['stage'] not in (1,2) or spec['role'] not in ('reference','trial')): raise ValueError('invalid sampling challenge')
    return spec['milestone'],spec['stage'],spec['role']
