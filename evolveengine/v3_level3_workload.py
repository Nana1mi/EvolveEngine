"""Versioned real-text input sampling for independent Level 3 timing."""
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re

from .v3_harness import canonical, digest

SCHEMA = 'level3-real-text-token-corpus-v1'
DISTRIBUTION = 'nonoverlapping_contiguous_real_text_v1'
MAX_CORPUS_BYTES = 32 << 20
NOISE_PAIRS = 6
FIRST_LIVE_PAIR = NOISE_PAIRS + 1
MAX_PAIRS = 512
EXTENDED_PAIRS = 1 << 31


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def load_corpus(path, pin, tokenizer_pin):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_CORPUS_BYTES + 1)
    _require(0 < len(raw) <= MAX_CORPUS_BYTES and digest(raw) == pin, 'timing corpus fingerprint mismatch')
    corpus = json.loads(raw)
    _require(raw == canonical(corpus) + b'\n' and type(corpus) is dict and
             set(corpus) == {'schema', 'tokenizer_sha256', 'documents'} and corpus['schema'] == SCHEMA and
             _sha(tokenizer_pin) and corpus['tokenizer_sha256'] == tokenizer_pin, 'timing corpus/tokenizer identity invalid')
    docs = corpus['documents']
    _require(type(docs) is list and 2 <= len(docs) <= 64, 'timing corpus requires diverse source documents')
    seen = set()
    for doc in docs:
        _require(type(doc) is dict and set(doc) == {'source', 'sha256', 'ids'} and _sha(doc['sha256']) and
                 doc['sha256'] not in seen and type(doc['source']) is str and 0 < len(doc['source']) <= 400 and
                 type(doc['ids']) is list and doc['ids'] and all(type(i) is int and 0 <= i < (1 << 31) for i in doc['ids']),
                 'invalid timing source or token sequence')
        seen.add(doc['sha256'])
    return [doc['ids'] for doc in docs]


def select_inputs(documents, scenes, waves, *, seed, pair, previously_used=frozenset()):
    """Use the same pair index on both arms. A fresh service gets only its own inputs."""
    _require(_sha(seed) and type(pair) is int and 0 <= pair < (1 << 31), 'invalid timing seed or pair index')
    rng = Random(int.from_bytes(sha256(canonical([DISTRIBUTION, seed, pair])).digest(), 'big'))
    slots = [(n, scene, wave, lane) for scene, spec in scenes.items() for wave in range(waves)
             for lane, (n, _) in enumerate(spec)]
    pool = [(i, 0, len(ids)) for i, ids in enumerate(documents)]
    items = {scene: [[None for _ in spec] for _ in range(waves)] for scene, spec in scenes.items()}
    hashes = set()
    for n, scene, wave, lane in sorted(slots, key=lambda slot: (-slot[0], *slot[1:])):
        for _ in range(128):
            eligible = [(i, end - start - n + 1) for i, (_, start, end) in enumerate(pool) if end - start >= n]
            count = sum(weight for _, weight in eligible)
            _require(count > 0, 'timing corpus cannot supply nonoverlapping request windows')
            choice = rng.randrange(count)
            for pool_index, weight in eligible:
                if choice < weight:
                    break
                choice -= weight
            doc, start, end = pool[pool_index]
            offset = start + choice
            tokens = documents[doc][offset:offset + n]
            pin = digest(canonical(tokens))
            if pin in hashes or pin in previously_used or len(set(tokens)) < min(16, n):
                continue
            hashes.add(pin)
            items[scene][wave][lane] = tokens
            pool[pool_index:pool_index + 1] = [(doc, start, offset), (doc, offset + n, end)]
            break
        else:
            raise ValueError('timing corpus cannot supply distinct nondegenerate inputs')
    return items


def input_fingerprints(inputs):
    return {scene: tuple(tuple(digest(canonical(ids)) for ids in wave) for wave in waves)
            for scene, waves in inputs.items()}


def profile_inputs(documents, scenes, count, seed, chat=None):
    """Separate traffic for publishable traces. Never profile any scored chat wave."""
    excluded = ({pin for waves in input_fingerprints(chat.waves).values() for wave in waves for pin in wave}
                if chat is not None else set())
    return select_inputs(documents, scenes, count, seed=digest(canonical(['profile-only-v1', seed])),
                         pair=0, previously_used=excluded)


class TimingSchedule:
    """Deterministic pair plans without reusing exact prompts across pair indices."""
    def __init__(self, documents, scenes, waves, seed, *, maximum=MAX_PAIRS):
        self.documents, self.scenes, self.waves, self.seed = documents, scenes, waves, seed
        self.maximum = maximum
        self.index, self.current, self.used = -1, None, set()

    def get(self, pair):
        _require(type(pair) is int and 0 <= pair < self.maximum, 'timing pair outside registered schedule')
        if pair < self.index:
            self.index, self.current, self.used = -1, None, set()
        while self.index < pair:
            index = self.index + 1
            inputs = select_inputs(self.documents, self.scenes, self.waves, seed=self.seed, pair=index,
                                   previously_used=self.used)
            hashes = input_fingerprints(inputs)
            self.used.update(pin for waves in hashes.values() for wave in waves for pin in wave)
            self.index, self.current = index, (inputs, hashes)
        return self.current
