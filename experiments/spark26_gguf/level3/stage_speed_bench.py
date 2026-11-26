"""Stage pinned NVIDIA SPEED-Bench splits for a standalone internal benchmark."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from evolveengine.v3_harness import canonical
from experiments.spark26_gguf.level3.intake_public_chat import file_digest, privacy_safe

DATASET = 'https://huggingface.co/datasets/nvidia/SPEED-Bench'
LICENSE = 'NVIDIA Evaluation Dataset License Agreement'
CATEGORIES = {'low_entropy','mixed','high_entropy'}
QUALITATIVE_CATEGORIES = {'coding','humanities','math','qa','rag','reasoning','stem','writing','multilingual',
                          'summarization','roleplay'}
SPLITS = {'qualitative','throughput_1k','throughput_2k','throughput_8k','throughput_16k','throughput_32k','throughput_100k'}


def stage(path, file_pin, output, revision, split):
    import pyarrow.parquet as pq

    if split not in SPLITS or file_digest(path)!=file_pin or len(revision)!=40:
        raise ValueError('invalid pinned SPEED-Bench snapshot')
    rows, seen, rejected = [], set(), Counter()
    categories = QUALITATIVE_CATEGORIES if split=='qualitative' else CATEGORIES
    counts = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['question_id','category','sub_category','turns']):
        for row in batch.to_pylist():
            identity, category, turns = (row[k] for k in ('question_id','category','turns'))
            if (type(identity) is not str or not identity or identity in seen or category not in categories
                    or type(turns) is not list or not turns or not all(type(t) is str and t.strip() for t in turns)):
                raise ValueError('unexpected SPEED-Bench record')
            seen.add(identity)
            if not privacy_safe([dict(content=turn) for turn in turns]):
                rejected[category] += 1
                continue
            counts[category] += 1
            rows.append(row)
    if set(counts)!=categories:
        raise ValueError('SPEED-Bench category coverage is incomplete')
    with output.open('xb') as handle:
        for row in rows:
            handle.write(canonical(row)+b'\n')
    receipt = dict(dataset=DATASET,revision=revision,split=split,parquet_sha256=file_pin,
                   jsonl_sha256=file_digest(output),counts=counts,privacy_format_rejected=rejected,
                   license=LICENSE,purpose='internal_speculation_benchmark',included_in_main_score=False,
                   pinned_sglang_single_turn_throughput=split.startswith('throughput_'))
    with output.with_suffix('.receipt.json').open('xb') as handle:
        handle.write(canonical(receipt)+b'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquet',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--split',choices=sorted(SPLITS),required=True)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(stage(args.parquet,args.sha256,args.output,args.revision,args.split),sort_keys=True))


if __name__ == '__main__':
    main()
