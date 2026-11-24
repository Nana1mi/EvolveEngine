"""Read-only audit of the frozen Level 3 registration. Never dispatches work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = 'experiments/spark26_gguf/level3/acceptance-v2.json'
MANIFEST_SHA256 = '4c029badae73d820f2a0a1175dfb4bbf3e1426e7b6e1919c7bf5b0ce06e56c20'
DOMAINS = dict(A=('A1', 'A2'), B=('B1', 'B2', 'B3'), C=('C1', 'C2', 'C3'),
               D=('D1', 'D2', 'D3'), E=('E1', 'E2'), F=('F1', 'F2'))
LAUNCH = dict(requires_all_domains=True, requires_separate_resource_authorization=True,
              performance_gain_required_for_engineering_acceptance=False, formal_w_or_b0_b2=False)


def _file(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if (name.is_absolute() or not name.parts or '..' in name.parts or '\\' in relative
            or ':' in relative or name.as_posix() != relative):
        raise ValueError(f'invalid registration path: {relative}')
    path = root.joinpath(*name.parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'not a file inside registration repository: {relative}')
    return path


def _verified_bytes(root: Path, relative: str, expected: str) -> bytes:
    data = _file(root, relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'frozen SHA-256 mismatch: {relative}')
    return data


def audit(root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    manifest = json.loads(_verified_bytes(root, MANIFEST, MANIFEST_SHA256))
    for key in ('objective', 'prior_registration'):
        item = manifest[key]
        _verified_bytes(root, item['path'], item['sha256'])
    if (manifest['protocol'] != 'spark26-level3-exploration-v2'
            or tuple(row['id'] for row in manifest['domains']) != tuple(DOMAINS)
            or manifest['launch'] != LAUNCH):
        raise ValueError('unexpected acceptance registration structure')
    domains = []
    for row in manifest['domains']:
        tests = row['acceptance_tests']
        if row['required'] is not True or tuple(tests) != DOMAINS[row['id']]:
            raise ValueError(f'missing required checks: {row["id"]}')
        for item in row['existing_evidence']:
            _file(root, item)
        domains.append(dict(id=row['id'], capability=row['capability'],
                            status_at_freeze=row['current_status'], checks=list(tests)))
    return dict(protocol=manifest['protocol'], audit_basis_git=manifest['audit_basis_git'],
                manifest_sha256=MANIFEST_SHA256, registration_valid=True, domains=domains,
                engineering_acceptance_evaluated=False, launch_admission_evaluated=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), indent=2))


if __name__ == '__main__':
    main()
