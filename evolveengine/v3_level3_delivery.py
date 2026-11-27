"""Export the Engine and Harness explicitly delivered by a settled Level 3 run."""
import argparse
from contextlib import closing
import json
from math import ulp
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from .schema import digest as object_digest
from . import v3_level3_scoring as business, v3_level3_weight_audit as weights
from .v3_harness import Lineage, canonical, digest
from .v3_level3_lifecycle import ROOT as LIFECYCLE
from .v3_level3_run import WorkerResult, compare_dev
from .v3_level3_source import SourceContract

MMQ_HEADER = 'python/sglang/kernels/aot/csrc/quantization/gguf/mmq.cuh'
NOTICE_DIR = Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf/level3/third-party'
NOTICE_FILES = ('THIRD_PARTY.md', 'vllm-4492e3a-LICENSE.txt', 'llama.cpp-b2899-LICENSE.txt')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _journal_event(db, key):
    row = db.execute('SELECT content FROM events WHERE id = ?', (key,)).fetchone()
    _need(row is not None, f'Missing lifecycle event: {key}')
    return json.loads(row[0])


def _source(db, result, contract, engine_id):
    source_ids = {event['source_object'] for event in result['events']
                  if event['operation'] == 'engine_registered' and event['engine'] == engine_id}
    _need(len(source_ids) == 1, 'Expected one registered source object for the chosen Engine')
    source_id, = source_ids
    row = db.execute('SELECT content FROM objects WHERE id = ? AND kind = ?',
                     (source_id, 'level3_engine')).fetchone()
    _need(row is not None, 'Missing chosen Engine source')
    source = json.loads(row[0])
    _need(object_digest({'kind': 'level3_engine', 'content': source}) == source_id,
          'Chosen Engine source object hash mismatch')
    engine = contract.from_args(source)
    _need(engine.digest == engine_id and contract.to_args(engine) == source, 'Chosen Engine source does not replay')
    return source


def _measurement(row):
    return WorkerResult(**{key: row[key] for key in
                           ('engine', 'operation', 'status', 'node_ms', 'build', 'evidence', 'data')})


def _same_measurement(replayed, archived):
    # libm can differ by a few binary rounding steps across Python builds.
    if type(replayed) is float:
        return type(archived) in (float, int) and abs(archived - replayed) <= 8 * ulp(replayed)
    if type(replayed) is dict:
        return type(archived) is dict and archived.keys() == replayed.keys() and all(
            _same_measurement(value, archived[key]) for key, value in replayed.items())
    if type(replayed) is list:
        return type(archived) is list and len(archived) == len(replayed) and all(
            _same_measurement(a, b) for a, b in zip(replayed, archived))
    return replayed == archived


def _sampling_proof(db, jobs, chosen, e0, milestone, report):
    """Replay the exact paid/charged draws from the immutable worker journal, not a marker in a nomination."""
    from . import v3_level3_sampling as sampling
    pins=report['receipts'];seen=set()
    _need(report['protocol']==sampling.VERSION and len(pins) in (2,4) and len(set(pins))==len(pins), 'invalid sampling receipt chain')
    for i,pin in enumerate(pins):
        # The settled result object is itself content-addressed by kind and content.
        job=jobs[pin];raw={k:job[k] for k in ('engine','operation','status','node_ms','build','evidence','data')}
        data=raw['data']; obj=object_digest(dict(kind='level3_worker_result',content=raw))
        row=db.execute('SELECT content FROM objects WHERE id=? AND kind=?',(obj,'level3_worker_result')).fetchone()
        role='trial' if i%2 else 'reference';stage=i//2+1
        _need(row and json.loads(row[0])==raw and raw['engine']==chosen['engine'] and raw['build']==chosen['build'] and
              raw['operation']=='develop_quality' and raw['status']=='sampled' and data['protocol']==sampling.VERSION and
              data['challenge']==dict(milestone=milestone,role=role,stage=stage) and data['service'] not in seen and
              data['observed_engine']==(chosen['engine'] if i%2 else e0), 'sampling receipt not bound to chosen build or E0')
        seen.add(data['service'])
        if i%2:
            _need(data['reference']==pins[i-1] and data['decision']==sampling.decision(data['test']) and
                  data['decision']==('inconclusive' if i<len(pins)-1 else report['decision']), 'unregistered sampling retest or disposition')
        else: _need(data['decision']=='reference','sampling reference missing')
    return report


def _verify(run, contract, proposal_segment=None):
    documents = {name: (run / name).read_bytes() for name in (
        'registration.json', 'settlement.json', 'run/result.json', 'performance-trajectory.json')}
    registration, settlement, result, trajectory = map(json.loads, documents.values())
    scoring = registration.get('business_scoring')
    _need(scoring == result.get('business_scoring') and
          (scoring is None or scoring == business.resolve_policy(
              business.VERSION if scoring['version'] == business.VERSION else scoring)),
          'Business scoring registration does not match the run')
    policy = registration.get('weight_audit')
    _need(policy == result.get('weight_audit') and policy in (None, *weights.POLICIES), 'weight review registration mismatch')
    staged = proposal_segment is not None
    if staged:
        _need(policy and type(proposal_segment) is int and proposal_segment > 0 and
              settlement['status'] == result['status'] == 'incomplete' and settlement['last_phase']=='worker_close',
              'stage export needs a settled paused audit run')
        proposals = [v for v in result.get('stage_proposals',[]) if v['segment']==proposal_segment]
        _need(len(proposals)==1 and proposals[0]['cold_verified'], 'no explicit cold-verified stage proposal')
        final = proposals[0]; lifecycle_key = f'segment/{proposal_segment}/proposal'; score = final['score']
    else:
        status, cause, lifecycle_key = (('pending_weight_review','agent_nomination_pending_review','weight_review_nomination')
                                        if policy else ('delivered','agent_delivery','delivery'))
        _need(settlement['status'] == result['status'] == status and settlement['last_phase'] == 'worker_close' and
              result['stop_cause'] == cause, 'Export requires a settled explicit Agent delivery')
        final = result['final']; score = result.get('chosen_business_score')
    chosen = final['chosen']
    _need(settlement['node_ms'] == result['node_ms'] == trajectory['charged_node_ms'] and
          settlement['worker_cost']['unsettled_node_reservation_ms'] == 0 and
          trajectory['reference'] == result['development_reference'], 'Worker accounting does not reconcile')
    jobs = {job['evidence']: job for job in result['jobs'] if 'evidence' in job}
    build = jobs.get(chosen['build'], {})
    quality = jobs.get(chosen['quality'], {})
    _need(build.get('operation') == 'develop_build' and build.get('status') == 'built' and
          build.get('engine') == chosen['engine'] and
          quality.get('operation') == 'develop_quality' and quality.get('status') == 'valid' and
          (quality.get('engine'), quality.get('build')) == (chosen['engine'], chosen['build']),
          'Chosen Engine, build and quality receipts do not match')
    if scoring:
        _need(build['data'].get('build_mode') == 'independent_cold', 'Business delivery requires a cold build')
    measurements = []
    database = (run / 'run/trajectory.sqlite3').resolve()
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        archived_policy = db.execute("SELECT content FROM events WHERE id = 'business_scoring/policy'").fetchone()
        _need((json.loads(archived_policy[0]) if archived_policy else None) == scoring,
              'Business scoring registration does not match the journal')
        journal_policy = db.execute("SELECT content FROM events WHERE id = 'weights/policy'").fetchone()
        _need((json.loads(journal_policy[0]) if journal_policy else None) == ({'policy':policy} if policy else None),
              'weight review policy differs from trusted journal')
        _need(_journal_event(db, LIFECYCLE + lifecycle_key) == final, 'Terminal delivery does not match lifecycle journal')
        _need(_journal_event(db, LIFECYCLE + 'quality/' + chosen['quality']) == chosen,
              'Chosen quality receipt does not match lifecycle journal')
        source = _source(db, result, contract, chosen['engine'])
        if policy:
            packets = [r['packet'] for (raw,) in db.execute("SELECT content FROM events WHERE id LIKE 'dispatch/%'")
                       if (r:=json.loads(raw)).get('operation')=='weight_audit_requested' and
                       r.get('trigger')=='final_nomination' and r.get('chosen')==chosen and r.get('segment')==final['segment']]
            _need(len(packets)==1, 'expected one trusted final audit request')
            folder=run/'run/weight-audit'; key=packets[0]; doc=weights.packet_read(folder,key)
            _need(doc['source_sha256']==digest(weights.wire(source)) and doc['context']['segment']==final['segment'],
                  'weight review does not bind final source and segment')
            if scoring and 'replay_audit_floors' in scoring:
                sample=_sampling_proof(db,jobs,chosen,contract.from_args({'files':{}}).digest,'final_nomination',doc['context']['sampling'])
                _need(sample['decision']=='pass','final sampling not passed')
                documents['sampling/final.json']=weights.wire(dict(screen=sample,observations=[jobs[k] for k in sample['receipts']]))
            report,_=weights.approved(folder,key,chosen)
            documents['weight-audit/packet.json']=weights.wire(doc)
            documents['weight-audit/review.json']=weights.wire(report)
        e0_id = contract.from_args({'files': {}}).digest
        for point in trajectory['points']:
            if (point['engine'], point['build']) != (chosen['engine'], chosen['build']):
                continue
            development = point['development']
            meter = jobs.get(development['receipt'], {})
            baseline = jobs.get(development['reference'], {})
            _need((meter.get('engine'), meter.get('build')) == (chosen['engine'], chosen['build']) and
                  baseline.get('engine') == e0_id, 'Development receipt does not match Engine or E0')
            replayed = compare_dev(_measurement(baseline), _measurement(meter),replay_floors=(scoring or {}).get('replay_audit_floors'))
            if scoring:
                replayed.update(scoring_version=scoring['version'],
                                weighted_score=business.weighted_score(replayed['scene_ratios'],
                                                                       scoring.get('weight_percent')))
                if scoring['version'] == business.WORKBUDDY_VERSION:
                    sla = business.workbuddy_sla_summary(meter['data'].get('scenes') or {})
                    if sla:
                        replayed['workbuddy_sla'] = sla
            if policy: replayed['gain_status'] = weights.PROVISIONAL
            _need(meter.get('performance') == development and _same_measurement(replayed, development),
                  'Development comparison does not replay')
            natural = scoring is not None and scoring['version'] in business.NATURAL_VERSIONS
            _need(meter['data']['timed_outputs_sha256'] == point['timed_outputs_sha256'] and
                  meter['data']['measurement_inputs_sha256'] == point['measurement_inputs_sha256'] and
                  meter['data']['measurement_pair'] == point['measurement_pair'] and
                  (all(row['comparable'] for row in replayed['output_work'].values()) if natural else
                   baseline['data']['timed_outputs_sha256'] == meter['data']['timed_outputs_sha256']),
                  'Paired input or greedy output evidence does not match')
            recorded = _journal_event(db, LIFECYCLE + 'measure/' + development['receipt'])
            measured_quality = recorded['quality']
            qualification = {**chosen, 'quality': measured_quality}
            _need(recorded == {**qualification, 'receipt': development['receipt']} and
                  _journal_event(db, LIFECYCLE + 'quality/' + measured_quality) == qualification,
                  'Development receipt does not match lifecycle journal')
            measurements.append(dict(point=point, baseline=baseline, candidate=meter, qualification=qualification))
        if scoring:
            matching = [row for row in result['jobs'] if (row.get('engine'), row.get('build')) ==
                        (chosen['engine'], chosen['build']) and 'performance' in row]
            _need([row['evidence'] for row in matching] ==
                  [m['candidate']['evidence'] for m in measurements], 'Business measurement coverage differs')
            durable_jobs = [json.loads(row[0]) for row in db.execute(
                "SELECT content FROM events WHERE id LIKE 'dispatch/%' ORDER BY id")]
            _need([r['evidence'] for r in durable_jobs if r['operation'] == 'worker_settled'] ==
                  [r['evidence'] for r in result['jobs'] if 'evidence' in r],
                  'Business measurement order differs from Worker settlements')
            _need(not any(r['operation'] in {'development_timed_output_mismatch','development_output_replay_mismatch',
                                           'development_workload_drift'} and r['engine'] == chosen['engine']
                          and jobs[r['candidate']]['build'] == chosen['build'] for r in durable_jobs),
                  'Chosen build has a timed output mismatch')
            summary = business.summarize_build(result['jobs'], chosen['engine'], chosen['build'],
                                               business.VERSION if scoring['version'] == business.VERSION else scoring)
            if scoring and 'replay_audit_floors' in scoring and 'historical_sampling' in score:
                hist=score['historical_sampling']
                if hist['decision']!='incomplete': _sampling_proof(db,jobs,chosen,e0_id,'historical_best',hist)
                _need(hist['decision']!='different','historical sampling rejected this build')
                summary['historical_sampling']=hist
            if policy: summary['gain_status']=weights.PROVISIONAL
            _need(summary['eligible_for_delivery'] and _same_measurement(summary, score),
                  'Chosen build does not meet the registered business gate')
    _need(measurements, 'No paired development evidence for the chosen build')
    lineage_root = run / 'run/harness-lineage'
    _need((lineage_root / 'versions').is_dir(), 'Missing archived Harness lineage')
    lineage = Lineage(lineage_root, registration['runtime'])
    snapshot = lineage.get(final['active_harness']) if staged else lineage.active()
    _need(snapshot.identity == final['active_harness'], 'Archived active Harness does not match delivery')
    return chosen, snapshot, source, build, quality, measurements, documents, registration['runtime'], score


def export_delivery(run, output, contract, *, proposal_segment=None):
    """Verify an existing run, then publish its bundle without rewriting the run."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    run = Path(run).resolve()
    chosen, snapshot, source, build, quality, measurements, documents, runtime, score = _verify(run, contract, proposal_segment)
    points = [measurement['point'] for measurement in measurements]
    source_bytes = canonical(source) + b'\n'
    manifest = dict(schema='level3-verified-delivery-v1', chosen=chosen, active_harness=snapshot.identity,
                    source_sha256=digest(source_bytes), pinned_e0=contract.baseline,
                    receipt_sha256={name: digest(raw) for name, raw in documents.items()},
                    paired_development=[dict(receipt=point['development']['receipt'],
                                             reference=point['development']['reference'],
                                             geomean=point['development']['equal_scene_geomean']) for point in points])
    if 'weight-audit/review.json' in documents:
        policy = json.loads(documents['registration.json']).get('weight_audit', weights.POLICY)
        manifest.update(weight_audit=policy, gain_status='externally_reviewed',
                        weight_review=json.loads(documents['weight-audit/review.json']))
    if 'sampling/final.json' in documents: manifest['independent_sampling_sha256']=digest(documents['sampling/final.json'])
    if score is not None:
        manifest.update(business_scoring=json.loads(documents['registration.json'])['business_scoring'],
                        business_score=(score | {'gain_status':'externally_reviewed'} if 'weight_review' in manifest else score))
        for measurement, row in zip(points, manifest['paired_development']):
            row['weighted_score'] = measurement['development']['weighted_score']
            row['scene_ratios'] = measurement['development']['scene_ratios']
    notices = {}
    if MMQ_HEADER in source['files']:
        notices = {name: (NOTICE_DIR / name).read_bytes().replace(b'\r\n', b'\n') for name in NOTICE_FILES}
        manifest['attributions'] = {name: digest(raw) for name, raw in notices.items()}
    reuse = []
    chosen_paths = set(source['files']) | set(source.get('deleted', ()))
    for path in sorted((run / 'web-research').glob('reuse-*.json')):
        row = json.loads(path.read_bytes())
        pin = digest(canonical(row))
        _need(path.stem == 'reuse-' + pin, 'research attribution receipt changed')
        matched = sorted(chosen_paths.intersection(row['paths']))
        if matched:
            reuse.append(dict(receipt=pin, delivered_paths=matched, **row))
    if reuse:
        manifest['research_attribution_sha256'] = digest(canonical(reuse)+b'\n')
    with TemporaryDirectory(prefix='.level3-export-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'delivery'
        staged.mkdir()
        if 'sampling/final.json' in documents:
            (staged/'sampling').mkdir(); (staged/'sampling/final.json').write_bytes(documents['sampling/final.json'])
        if 'weight-audit/review.json' in documents:
            proof=manifest['weight_review']['runtime_evidence_sha256']
            archived=(run/'run/weight-audit/runtime-evidence'/proof).read_bytes()
            _need(digest(archived)==proof,'runtime evidence changed during export')
            (staged/'weight-audit').mkdir()
            (staged/'weight-audit/runtime-witness.json').write_bytes(archived)
            for name in ('weight-audit/packet.json','weight-audit/review.json'):
                (staged/name).parent.mkdir(exist_ok=True)
                (staged/name).write_bytes(documents[name])
        (staged / 'engine-source.json').write_bytes(source_bytes)
        (staged / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
        (staged / 'evidence.json').write_bytes(canonical(dict(build=build, quality=quality, measurements=measurements)) + b'\n')
        if notices:
            (staged / 'third-party').mkdir()
            for name, raw in notices.items():
                (staged / 'third-party' / name).write_bytes(raw)
        if reuse:
            (staged / 'research-attribution.json').write_bytes(canonical(reuse)+b'\n')
        harness = staged / 'harness'
        harness.mkdir()
        snapshot.stage(harness / 'files', runtime)
        (harness / 'manifest.json').write_bytes(snapshot.manifest_bytes)
        staged.rename(output)
    return manifest


def contract_for_run(run, archive):
    reg = json.loads((Path(run) / 'registration.json').read_bytes())
    if reg.get('weight_audit') == weights.BF16_POLICY:
        return SourceContract.acquire_spark26_bf16_e0(archive)
    return SourceContract.acquire_spark26_e0(archive, Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--proposal-segment',type=int,help='export this explicit, reviewed cold-build nomination of a paused continuous run')
    opts = parser.parse_args()
    contract = contract_for_run(opts.run, opts.archive)
    print(canonical(export_delivery(opts.run, opts.output, contract, proposal_segment=opts.proposal_segment)).decode())


if __name__ == '__main__':
    main()
