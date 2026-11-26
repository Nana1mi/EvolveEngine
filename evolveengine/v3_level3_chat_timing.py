"""Pinned chat request pools shared by the E0 and candidate observers."""
import json
from pathlib import Path
from random import Random

from .v3_harness import canonical, digest
from .v3_level3_workload import MAX_CORPUS_BYTES, MAX_PAIRS, _require, _sha, input_fingerprints

SCHEMA = 'level3-thinking-chat-requests-v1'
SPLITS = {'development', 'generalization', 'holdout'}
W3_INPUT_RANGE = (3990, 4202)


GENERALIZATION = 'paired-short-long-v1'


def bounds(scene, nominal, split, wave):
    if split != 'generalization': return W3_INPUT_RANGE if scene=='W3' else (nominal,nominal)
    lo,hi = W3_INPUT_RANGE if scene=='W3' else (3*nominal//4,5*nominal//4)
    return (lo,nominal-1) if wave%2==0 else (nominal+1,hi)


def length_map(inputs):
    return {s:tuple(tuple(map(len,w)) for w in waves) for s,waves in inputs.items()}


def load_chat(ref, tokenizer_pin, template_pin, scenes, warm, timed, *, split='development', engine=None):
    _require(type(ref) is dict and set(ref) == {'path', 'sha256'} and _sha(ref['sha256']),
             'invalid chat workload reference')
    path = Path(ref['path'])
    _require(path.is_absolute() and path.resolve(strict=True) == path, 'unsafe chat workload path')
    with path.open('rb') as handle:
        raw = handle.read(MAX_CORPUS_BYTES+1)
    _require(0 < len(raw) <= MAX_CORPUS_BYTES and digest(raw) == ref['sha256'], 'chat workload hash mismatch')
    doc = json.loads(raw)
    _require(raw == canonical(doc)+b'\n' and set(doc) == {
        'schema','split','enable_thinking','tokenizer_sha256','template_sha256','screening','requests'},
        'invalid frozen chat workload')
    _require(doc['schema'] == SCHEMA and doc['split'] == split and split in SPLITS
             and doc['enable_thinking'] is True and doc['tokenizer_sha256'] == tokenizer_pin
             and doc['template_sha256'] == template_pin, 'chat workload model or split differs')
    report = doc['screening']
    if split != 'generalization':
        _require(type(report) is dict and set(report) == {'engine','receipt_sha256'}
                 and _sha(report['engine']) and _sha(report['receipt_sha256'])
                 and (engine is None or report['engine'] == engine), 'missing pinned E0 screen')
    else:
        _require(report is None, 'generalization uses unscreened requests')
    _require(type(doc['requests']) is dict and doc['requests'].keys() == scenes.keys(), 'chat scene coverage differs')
    waves = {}
    seen_tokens, seen_sources = set(), set()
    for scene, shape in scenes.items():
        rows = doc['requests'][scene]
        _require(type(rows) is list and len(rows) >= 2*warm+timed, 'insufficient charted chat requests')
        if split=='generalization': _require(timed>=2,'both timed length strata required')
        waves[scene] = []
        for wave_index,wave in enumerate(rows):
            _require(type(wave) is list and len(wave) == len(shape), 'chat request wave width differs')
            tokens = []
            for row, (input_len, _) in zip(wave,shape):
                _require(type(row) is dict and set(row) == {'ids','source','license','source_id','task'} and
                         all(type(row[k]) is str and 0 < len(row[k]) <= 512 for k in ('source','license','source_id','task')),
                         'missing chat request provenance')
                ids = row['ids']
                lower, upper = bounds(scene,input_len,split,wave_index)
                _require(type(ids) is list and lower <= len(ids) <= upper and
                         all(type(i) is int and 0 <= i < 131072 for i in ids), 'chat request input length differs')
                key = (row['source'],row['source_id'])
                pin = digest(canonical(ids))
                _require(pin not in seen_tokens and key not in seen_sources, 'repeated chat request')
                seen_tokens.add(pin)
                seen_sources.add(key)
                tokens.append(ids)
            waves[scene].append(tokens)
    return waves


class ChatSchedule:
    """Each matched pair uses the same frozen timed set in a pair-specific order."""
    def __init__(self, waves, warm, timed, seed):
        self.waves, self.warm, self.timed, self.seed = waves, warm, timed, seed
        self.maximum = MAX_PAIRS

    def get(self, pair, *, replay=False):
        _require(type(pair) is int and 0 <= pair < self.maximum, 'timing pair outside registered schedule')
        inputs = {}
        for scene, waves in self.waves.items():
            end = len(waves) if scene == 'W3' else 2*self.warm+self.timed
            order = list(waves[2*self.warm:end])
            rng = Random(int(digest(canonical([SCHEMA,self.seed,scene,pair])),16))
            rng.shuffle(order)
            warm = waves[self.warm:2*self.warm] if replay else waves[:self.warm]
            inputs[scene] = warm + (list(reversed(order)) if replay else order)
        return inputs, input_fingerprints(inputs)
