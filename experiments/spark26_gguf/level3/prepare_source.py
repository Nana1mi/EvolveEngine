"""Prepare a complete, hash-verified Spark GGUF SGLang E0 tree offline."""
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_source import SourceContract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='local sglang.tar.gz matching the E0 lock')
    parser.add_argument('destination', type=Path, help='new path under an existing trusted directory')
    parser.add_argument('--manifest', type=Path, help='new file outside the source tree')
    args = parser.parse_args()
    if args.manifest:
        dest, manifest = args.destination.resolve(), args.manifest.resolve()
        if manifest.is_relative_to(dest) or args.manifest.exists() or args.manifest.is_symlink():
            parser.error('manifest must be a fresh path outside the destination')
    contract = SourceContract.acquire_spark26_e0(args.archive, ROOT)
    engine = contract.from_args({'files': {}})
    contract.materialize(engine, args.destination)
    if args.manifest:
        with args.manifest.open('xb') as f:
            f.write(canonical(contract.inventory_manifest()) + b'\n')
    print(f'baseline={contract.baseline} engine={engine.digest} files={len(contract.base)}')


if __name__ == '__main__':
    main()
