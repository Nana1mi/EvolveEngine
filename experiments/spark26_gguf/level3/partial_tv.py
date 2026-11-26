"""TV evidence from partial distributions; unobserved mass is never filled with zeros.

For any set of token coordinates observed in both normalized distributions,
the unknown remainder can overlap anywhere between complete overlap of its
smaller mass and disjoint support. A partial view can certify a bound or leave
it inconclusive; it cannot force exact token equality or waive EOS checks.
"""
from math import exp, fsum, isfinite


def matched_tv(left, right, vocab, *, normalization_error=0.0):
    """Return conservative TV lower/upper bounds, or reject bad wire values.

    Each view is a map of token IDs to log probabilities. `normalization_error`
    is a registered uncertainty on each complete vector's total mass, not a
    candidate-tuned allowance. Mass outside the intersection is unknown even
    when the token was present in exactly one arm's top list.
    """
    if type(vocab) is not int or vocab<=0 or type(normalization_error) not in (int,float) or not 0<=normalization_error<1:
        raise ValueError('invalid probability comparison policy')
    for side in (left,right):
        if (type(side) is not dict or not side or any(type(t) is not int or not 0<=t<vocab or
                type(v) not in (int,float) or not isfinite(v) or v>0 for t,v in side.items())):
            raise ValueError('invalid partial log probability view')
        if fsum(exp(v) for v in side.values())>1+normalization_error:
            raise ValueError('partial vector exceeds normalization')
    shared=left.keys() & right.keys()
    if not shared: raise ValueError('no common probability coordinates')
    # Stable ordering also makes JSON summaries reproducible across processes.
    ids=sorted(shared); p=[exp(left[t]) for t in ids]; q=[exp(right[t]) for t in ids]
    pm,qm=fsum(p),fsum(q); observed=fsum(abs(a-b) for a,b in zip(p,q,strict=True))
    if len(shared)==vocab:
        if abs(pm-1)>normalization_error or abs(qm-1)>normalization_error:
            raise ValueError('complete vector is not normalized')
        rem=lower_rem=0.
    else:
        # Conservative remainder-mass intervals that include rounding slack.
        pl,ph=max(0.,1-normalization_error-pm),max(0.,1+normalization_error-pm)
        ql,qh=max(0.,1-normalization_error-qm),max(0.,1+normalization_error-qm)
        lower_rem=max(0.,pl-qh,ql-ph)
        rem=max(abs(ph-ql),abs(qh-pl)) if len(shared)==vocab-1 else ph+qh
    return dict(lower=(observed+lower_rem)/2,upper=(observed+rem)/2,shared=len(shared),
                observed_mass=[pm,qm],complete=len(shared)==vocab)


def certify(report, maximum):
    if (type(maximum) not in (int,float) or not 0<=maximum<=1 or
        not all(type(report[k]) in (int,float) and isfinite(report[k]) for k in ('lower','upper')) or
        not 0<=report['lower']<=report['upper']):
        raise ValueError('invalid TV interval or boundary')
    if report['upper']<=maximum: return 'certified'
    if report['lower']>maximum: return 'different'
    return 'inconclusive'
