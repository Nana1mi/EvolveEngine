"""Offline P1 release gate. No model calls, no inferred reviewer identity.

A real independent reviewer must attest to the replay and issue this manifest. Its
SHA-256 is obtained out-of-band; this code checks binding/criteria, not identity.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from evolveengine.schema import digest, safe_path
from .trial import LIMIT


class NotReleased(ValueError):
    pass


def require(ok, msg):
    if not ok:
        raise NotReleased(msg)


def is_hash(x):
    return isinstance(x, str) and bool(re.fullmatch(r'[a-f0-9]{64}', x))


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def read_registered(base, name, commitment):
    require(safe_path(name) and is_hash(commitment), 'Invalid evidence file reference')
    root = Path(base).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Evidence outside review bundle')
    data = path.read_bytes()
    require(sha256(data).hexdigest() == commitment, 'Evidence commitment mismatch')
    return data


def verified_cards(manifest_bytes, trusted_sha, evidence_root, expected):
    """Expected = controller's frozen inputs, never populated from the reviewer manifest itself.

    Includes card IDs/roles/digests and the *minimum* preregistered noise line. The
    reviewed receipts are independent outputs; no author summary is accepted here.
    """
    require(is_hash(trusted_sha) and sha256(manifest_bytes).hexdigest() == trusted_sha, 'Untrusted release')
    m = json.loads(manifest_bytes)
    require(m.get('schema') == 'p1-independent-release-v1' and m.get('status') == 'approved', 'No approved release')
    require(isinstance(m.get('reviewer'),str) and m['reviewer'].strip() and isinstance(m.get('author'),str)
            and m['author'].strip() and m['author'].strip().casefold() != m['reviewer'].strip().casefold()
            and m.get('reviewer_did_not_build') is True, 'Missing independent reviewer')
    require(m.get('expected_digest') == digest(expected), 'Reviewer attested a different protocol')
    for key in ('engine_sha','harness_sha','worker_digest','executor_protocol_sha256'):
        require(expected[key] == m.get(key), f'{key} differs from frozen protocol')
    require(m.get('budget') == LIMIT.record(), 'Unequal or missing trial budget')
    cards = expected['cards']; items = m.get('cards', {})
    require(len(cards)==4 and sorted(c['kind'] for c in cards.values()) == ['negative','negative','positive_cross','positive_local']
            and isinstance(items,dict) and set(items)==set(cards), 'Incomplete four-card set')
    for k, spec in cards.items():
        item=items[k]; r=json.loads(read_registered(evidence_root,item['receipt_file'],item['receipt_sha256']))
        require(r.get('schema')=='p1-independent-replay-v1' and r.get('status')=='accepted'
                and r.get('reviewer')==m['reviewer'] and r.get('fresh_checkout') is True, 'Missing independent replay')
        require(all(r.get(f)==spec[f] for f in ('kind','card_sha256','fixture_sha256','inputs_sha256'))
                and r.get('card_id')==k and r.get('engine_sha')==expected['engine_sha']
                and r.get('worker_digest')==expected['worker_digest'], 'Wrong card/fixture/deployment')
        raw=json.loads(read_registered(evidence_root,r['raw_file'],r['raw_sha256']))
        require(raw.get('card')==k and raw.get('source')=='independent_node_replay'
                and raw.get('worker_digest')==expected['worker_digest'] and raw.get('engine_sha')==expected['engine_sha']
                and raw.get('fixture_sha256')==spec['fixture_sha256'] and raw.get('inputs_sha256')==spec['inputs_sha256'],
                'Replay evidence not bound to this card')
        sources=raw.get('source_files',{})
        require(isinstance(sources,dict) and len(sources)>=(2 if spec['kind']=='negative' else 8),
                'Original independent node receipts missing')
        for filename, commitment in sources.items():
            read_registered(evidence_root, filename, commitment)
        if spec['kind']=='negative':
            require(raw.get('observed')==spec['witness_code'] and raw.get('detection_confirmed') is True,
                    'Negative control not reproduced')
        else:
            qc=raw.get('quality',{}); checks={'launch','full_vocab','kl','top1','sequence','output_length','eos','kv_isolation'}
            require(isinstance(qc,dict) and set(qc)==checks and all(v is True for v in qc.values()) and raw.get('mechanism_verified') is True,
                    'Mechanism or quality not confirmed')
            line=raw.get('noise_log'); pairs=raw.get('pairs',[])
            require(finite(line) and line >= expected['minimum_noise_log'] > 0 and isinstance(pairs,list)
                    and len(pairs)==len(spec['blocks'])==3 and [v.get('block_sha256') for v in pairs]==spec['blocks'],
                    'Invalid independent calibration or blocks')
            for v in pairs:
                a,b=v.get('e0_tok_s'),v.get('ref_tok_s')
                require(finite(a) and finite(b) and min(a,b)>0 and math.log(b/a)>line, 'Independent gain did not clear noise')
    # Return just IDs; caller must not pass this manifest or its evidence to Pi.
    return tuple(sorted(cards))


def balanced_schedule(ids):
    require(len(set(ids))==len(ids)==4,'Requires four distinct signed cards')
    # One new checkout per row. Alternate arm order per card; reverse it in repeat 2.
    return [(k,arm,rep+1) for rep in range(2) for i,k in enumerate(sorted(ids))
            for arm in (('single','short') if (i+rep)%2==0 else ('short','single'))]
