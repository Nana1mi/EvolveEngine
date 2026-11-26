"""Read settled development quality observations without dispatching any Worker calls."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from evolveengine.schema import canonical, digest
from evolveengine.v3_level3_collect import need
from evolveengine.v3_level3_trajectory_lane import SPLIT_PROTOCOL, trajectory_cases


KIND = 'level3_worker_result'


def observed(db):
    """Read a single snapshot; match each settled result to its reservation."""
    queries = db.execute('''SELECT content FROM events WHERE substr(id,1,9)='dispatch/' ORDER BY ordinal''')
    pending = None
    for (raw,) in queries:
        event = json.loads(raw)
        operation = event['operation']
        if operation == 'worker_reserved':
            need(pending is None, 'missing Worker settlement')
            pending = event
        elif operation == 'worker_uncertain':
            need(pending is not None and (pending['engine'], pending['worker_operation']) ==
                 (event['engine'], event['worker_operation']), 'unbound uncertain Worker operation')
            pending = None
        elif operation == 'worker_settled':
            oid = event['result_object']
            saved = db.execute('SELECT content FROM objects WHERE id=? AND kind=?', (oid, KIND)).fetchone()
            need(saved is not None, 'missing settled Worker object')
            result = json.loads(saved[0])
            need(digest(dict(kind=KIND, content=result)) == oid and event['evidence'] == result['evidence'],
                 'Worker object failed content validation')
            need(pending is not None and (pending['engine'], pending['worker_operation']) ==
                 (result['engine'], result['operation']), 'unbound settled Worker operation')
            if result['operation'] == 'develop_quality' and 'sampling_challenge' not in pending:
                yield result
            pending = None


def summarize(results):
    groups, used = {}, set()
    for result in results:
        pin = result['evidence']
        if pin in used:
            continue
        used.add(pin)
        engine, build = result['engine'], result['build']
        need(build is not None, 'quality evidence must identify the tested build')
        data = result['data']
        record = data.get('trajectory_screen')
        receipt = dict(evidence=pin, status=result['status'], cases=data.get('case_outcomes', {}),
                       witness='unavailable' if record is None else 'observed')
        if record is not None:
            need(record['engine'] == engine and all(receipt['cases'][k] == v
                     for k, v in trajectory_cases(record, SPLIT_PROTOCOL).items()), 'mismatched quality witnesses')
            receipt.update(service=record['service'],
                           coverage_gaps={k: v['insufficient'] for k, v in record['groups'].items()})
        groups.setdefault((engine, build), []).append(receipt)

    out = []
    for (engine, build), rows in groups.items():
        # The same service is not an independent observation even if wrapped in two receipts.
        services = {r['service'] for r in rows if r['witness'] == 'observed'}
        witnessed = [r for r in rows if r['witness'] == 'observed']
        if not any(r['status'] == 'valid' for r in rows) or len(rows) < 2:
            continue
        signatures = {canonical((r['status'], r['cases'])) for r in witnessed}
        independent = len(services) == len(witnessed)
        if independent and len(services) >= 2 and len(signatures) > 1:
            status = 'quality_flip_review'
        elif not independent or any(r['witness'] == 'unavailable' or r['status'] == 'insufficient_evidence' for r in rows):
            status = 'coverage_review'
        else:
            status = 'consistent_observations'
        out.append(dict(engine=engine, build=build, status=status, observations=rows))
    return dict(schema='spark26-quality-stability-audit-v1', scoring=False, retry=False, repeated_builds=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path, help='trusted controller trajectory.sqlite3')
    args = parser.parse_args()
    uri = args.journal.resolve(strict=True).as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute('BEGIN')
        print(canonical(summarize(observed(db))))


if __name__ == '__main__':
    main()
