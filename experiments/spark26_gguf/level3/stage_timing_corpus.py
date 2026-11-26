"""Freeze tokenized public timing text against the exact target tokenizer."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_workload import MAX_CORPUS_BYTES, SCHEMA, load_corpus


def pinned(path, sha, cap):
    with Path(path).open('rb') as handle:
        body = handle.read(cap + 1)
    if not 0 < len(body) <= cap or digest(body) != sha:
        raise ValueError(f'input fingerprint or size mismatch: {Path(path).name}')
    return body


def stage(output, tokenizer, tokenizer_pin, documents):
    from tokenizers import Tokenizer

    codec = Tokenizer.from_str(pinned(tokenizer, tokenizer_pin, MAX_CORPUS_BYTES).decode('utf-8'))
    codec.no_padding()
    codec.no_truncation()
    if not 2 <= len(documents) <= 64:
        raise ValueError('register 2 to 64 timing documents')
    rows = []
    for path, pin, source in documents:
        text = pinned(path, pin, 4 << 20).decode('utf-8')
        if '\x00' in text or not text.strip() or not 0 < len(source) <= 400 or pin in {row['sha256'] for row in rows}:
            raise ValueError('invalid or repeated timing source')
        ids = codec.encode(text, add_special_tokens=False).ids
        if not ids:
            raise ValueError('source has no tokens')
        rows.append(dict(source=source, sha256=pin, ids=ids))
    raw = canonical(dict(schema=SCHEMA, tokenizer_sha256=tokenizer_pin, documents=rows)) + b'\n'
    if len(raw) > MAX_CORPUS_BYTES:
        raise ValueError('timing corpus too large')
    with output.open('xb') as handle:
        handle.write(raw)
    pin = digest(raw)
    load_corpus(output, pin, tokenizer_pin)
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--document', nargs=3, action='append', required=True, metavar=('PATH', 'SHA256', 'SOURCE'))
    args = parser.parse_args()
    print(stage(args.output, args.tokenizer, args.tokenizer_sha256, args.document))


if __name__ == '__main__':
    main()
