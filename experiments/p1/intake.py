"""Inspect a returned P1 review without reserving a trial, contacting a model or running a GPU."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

from .dispatch import protocol_hash
from .release import finite, require, verified_cards


def _utc(value):
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None, 'Timestamp lacks timezone')
    return stamp


def _json(root, name):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), 'Run summary outside returned directory')
    return json.loads(path.read_bytes())


def replay_cost(root, reviewer):
    """The parent reservation covers both sequential stages; do not add nested costs twice."""
    root = Path(root).resolve(strict=True)
    run = _json(root, 'run.json')
    require(run.get('schema') == 'p1-review-run-v1' and run.get('status') == 'ready_for_reviewer_decision'
            and run.get('reviewer_supplied') == reviewer, 'No matching completed independent run')
    begin, end = _utc(run['started_utc']), _utc(run['ended_utc'])
    wall = run['wall_seconds']
    require(finite(wall) and wall > 0 and end > begin, 'Invalid replay cost or time range')
    last = begin
    inside = 0.0
    for file, expected_status in [('negative-raw/summary.json', 'observed_not_independently_accepted'),
                                  ('round2-raw/summary.json', 'complete_author_only')]:
        # The frozen workers never sign cards, so their stage labels remain unchanged on a peer replay.
        stage = _json(root, file)
        a, b = _utc(stage['start_utc']), _utc(stage['end_utc'])
        seconds = stage['charged_seconds']
        require(stage.get('status') == expected_status and last <= a < b <= end,
                'Stage missing, overlapping, or from a different run')
        require(finite(seconds) and seconds > 0, 'Invalid stage charge')
        inside += seconds
        last = b
    return math.ceil(max(wall, (end - begin).total_seconds(), inside))


def inspect_review(expected, manifest, trusted_sha, root):
    require(expected.get('executor_protocol_sha256') == protocol_hash(), 'Frozen executor differs')
    cards = verified_cards(manifest, trusted_sha, root, expected)
    reviewer = json.loads(manifest)['reviewer']
    seconds = replay_cost(root, reviewer)
    return {'status': 'evidence_checked_no_dispatch', 'reviewer': reviewer, 'cards': list(cards),
            'review_run_seconds': seconds, 'within_8h_alone': seconds <= 8 * 3600,
            'total_study_budget_remaining': None, 'started_sessions': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', type=Path, required=True, help='controller-side frozen expected.json')
    parser.add_argument('--manifest', type=Path, required=True, help='final manifest returned by reviewer')
    parser.add_argument('--trusted-sha256', required=True, help='manifest SHA-256 confirmed out of band')
    parser.add_argument('--evidence', type=Path, required=True, help='returned independent run directory')
    args = parser.parse_args(argv)
    try:
        result = inspect_review(json.loads(args.expected.read_bytes()), args.manifest.read_bytes(),
                                args.trusted_sha256, args.evidence)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print('NOT READY: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
