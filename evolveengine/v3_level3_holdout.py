"""Host-only release gate for exploratory Q/T. No Agent RPC owns this module.

The designated custodian pins the review file out-of-band; hashes do not prove
who reviewed it. Raw Q/T and evaluator outputs remain in the custodian's lane.
"""
from hashlib import sha256
import json
import re

from .v3_level3_lifecycle import QualifiedEngine, ROOT

PREFIX = 'sealed_qt/'
SHA = re.compile(r'[0-9a-f]{64}\Z')


def _need(value, why):
    if not value:
        raise ValueError(why)


def _sha(value):
    return type(value) is str and SHA.fullmatch(value) is not None


def _label(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9_.:@/-]{1,160}',value) is not None


def review_metadata(review_bytes, *, approved_sha256):
    """Verify the metadata bytes without accessing private Q/T."""
    _need(type(review_bytes) is bytes and 0 < len(review_bytes) <= 16*1024 and _sha(approved_sha256)
          and sha256(review_bytes).hexdigest() == approved_sha256, 'review hash mismatch')
    row = json.loads(review_bytes)
    keys = {'scope','dataset_sha256','judge_sha256','custodian','user_review_sha256',
            'independent_model','model_review_sha256','access_policy_sha256'}
    _need(type(row) is dict and set(row)==keys and row['scope']=='level3_exploratory_qt'
          and all(_sha(row[k]) for k in keys if k.endswith('_sha256'))
          and _label(row['custodian']) and _label(row['independent_model'])
          and row['user_review_sha256'] != row['model_review_sha256'], 'missing independent review or custody')
    return row


def register_review(journal, review_bytes, *, approved_sha256):
    """Register only metadata before any session, using a custodian-pinned approval."""
    row = review_metadata(review_bytes, approved_sha256=approved_sha256)
    with journal.transaction():
        _need(journal.get_event(ROOT+'segment/1/start') is None, 'holdout approval must precede the first session')
        journal.event(PREFIX+'locked',dict(review_sha256=approved_sha256,metadata=row))
    return approved_sha256


def release_once(journal, *, approved_sha256, open_private, evaluate_private, verify_delivery):
    """Claim a one-shot evaluation after the actual terminal delivery event.

    `open_private` and `evaluate_private` are trusted custodian callbacks. No
    scores go into the exploration result. Failures after the claim are durable.
    """
    with journal.transaction():
        registration = journal.get_event(PREFIX+'locked')
        _need(registration is not None and registration['review_sha256']==approved_sha256, 'no matching holdout approval')
        delivered = journal.get_event(ROOT+'delivery')
        _need(delivered is not None and delivered.get('kind')=='delivered' and type(delivered.get('segment')) is int,
              'no final Agent delivery')
        _need(journal.get_event(ROOT+f"segment/{delivered['segment']}/end") == delivered
              and journal.get_event(ROOT+f"segment/{delivered['segment']+1}/start") is None,
              'delivery is not a terminal sealed segment')
        engine = QualifiedEngine(**delivered['chosen'])
        _need(_sha(delivered.get('active_harness')) and verify_delivery(engine) is True, 'delivered artifact not verified')
        _need(journal.get_event(PREFIX+'claimed') is None, 'holdout release already attempted')
        journal.event(PREFIX+'claimed',dict(review_sha256=approved_sha256,engine=engine.engine,segment=delivered['segment']))
    try:
        meta = registration['metadata']
        raw, judge = open_private(meta['custodian'])
        _need(type(raw) is bytes and type(judge) is bytes and sha256(raw).hexdigest()==meta['dataset_sha256']
              and sha256(judge).hexdigest()==meta['judge_sha256'], 'private Q/T payload drift')
        pointer = evaluate_private(engine,raw,judge)
        _need(_sha(pointer), 'private evaluator did not seal its result')
    except Exception as exc:
        journal.event(PREFIX+'failed',dict(error=type(exc).__name__))
        raise
    journal.event(PREFIX+'settled',dict(receipt_sha256=pointer))
    return pointer
