import concurrent.futures
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from evolveengine import observatory as api
from evolveengine import observatory_data as projection
from evolveengine.journal import Journal
from evolveengine.schema import canonical

E0, E1, E2 = '0'*64, '1'*64, '2'*64
R0, R1, R2 = '3'*64, '4'*64, '5'*64
H0, H1, CP, B1, Q1 = ('a'*64, 'b'*64, 'c'*64, 'd'*64, 'e'*64)
SECRET = 'private marker never exposed'
REGISTRATION = {'scope': 'one_engineering_exploration_without_holdout',
                'holdout_review_sha256': None, 'full_acceptance': False}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding='utf-8')


def ev(db, key, op, **data):
    db.event(key, {'operation': op, **data})


def work(root, db, number, op, engine, proof, status, *, corrupt=False, stale_transport=False):
    key = f'w/{number}'
    ev(db, key+'/start', 'worker_reserved', engine=engine, worker_operation=op, reserved_ms=100)
    row = dict(operation=op, status=status, engine=engine, node_ms=19, evidence=proof, build=B1,
               data=dict(scenes={'W1': {'tpot_ms': 9 if engine == E0 else 6, 'unlisted field': SECRET}},
                         candidate_output=SECRET))
    address = db.put('level3_worker_result', row)
    ev(db, key+'/end', 'worker_settled', evidence=proof, result_object=address)
    if corrupt:
        db.db.execute('UPDATE objects SET content=? WHERE id=?', (canonical(dict(row, node_ms=1)), address))
    prefix = f'{number:05}'
    write(root/'worker-operations'/f'{prefix}-reserved.json',
          dict(engine=engine, operation=op, cap_ms=100, elapsed_ms=20, settled_input=123,
               api_cost_reservation_ledger=dict(total_charged_rmb='0.011')))
    write(root/'worker-operations'/f'{prefix}-transport.json',
          dict(evidence=proof, result_sha256='f'*64 if stale_transport else projection.sha_bytes(canonical(row).encode()),
               elapsed_ms=40))


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.db = self.root/'run'/'trajectory.sqlite3'
        write(self.root/'registration.json', dict(REGISTRATION, prompt=SECRET))

    def snapshot(self):
        result = projection.local_run(self.root, 'example')
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn(str(self.root), json.dumps(result))
        return result

    def test_only_bound_comparisons_get_points_and_clocks(self):
        with Journal(self.db) as db:
            work(self.root, db, 1, 'develop_measure', E0, R0, 'observed')
            ev(db, 'reference', 'development_reference', evidence=R0, engine=E0)
            work(self.root, db, 2, 'develop_measure', E1, R1, 'observed', stale_transport=True)
            work(self.root, db, 3, 'develop_measure', E2, R2, 'observed', corrupt=True)
            work(self.root, db, 4, 'develop_build', E2, 'f'*64, 'build_failed')
            for i, engine, receipt in ((1, E1, R1), (2, E2, R2)):
                ev(db, f'compare/{i}', 'development_comparison', engine=engine, receipt=receipt,
                   reference=R0, equal_scene_geomean=1.5, scene_ratios={'W1': 1.5})
            db.event('level3/delivery', dict(chosen=dict(engine=E1, build=B1, quality=Q1)))
            s = self.snapshot()  # the writer is still open; the sidecar must not take its lease
        self.assertEqual(s['baseline'], E0)
        self.assertEqual([p['id'] for p in s['points']], [R1])
        p = s['points'][0]
        self.assertEqual(p['reference_scenes']['W1'], {'tpot_ms': 9})
        self.assertEqual(p['resources']['input_tokens'], 123)
        self.assertNotIn('wall_ms', p['resources'])   # wrong result hash, despite matching receipt id
        self.assertIsNone(s['delivery']['point_id'])
        self.assertEqual(s['delivery']['evidence'], Q1)
        self.assertTrue(any(e.get('failure') and e.get('status') == 'build_failed' for e in s['events']))
        self.assertTrue(any('1 unbound' in n for n in s['notices']))

    def test_operator_measurement_is_bound_distinct_and_idempotent(self):
        approval = 'e'*64
        write(self.root/'registration.json', dict(REGISTRATION, approval_sha256=approval))
        write(self.root/'settlement.json', dict(status='incomplete', elapsed_ms=100))
        write(self.root/'run'/'result.json', dict(stop_cause='operator_pause_after_segment'))
        with Journal(self.db) as db:
            work(self.root, db, 1, 'develop_measure', E0, R0, 'observed')
            ev(db, 'reference', 'development_reference', evidence=R0, engine=E0)
            ev(db, 'candidate', 'engine_registered', engine=E1, parent=E0)
            work(self.root, db, 2, 'develop_measure', E1, R1, 'observed')
            ev(db, 'comparison', 'development_comparison', engine=E1, receipt=R1,
               reference=R0, weighted_score=1.02, scene_ratios={'W1': 1.02})
        row = dict(engine=E1, reference_engine=E0, timing=R2, build=B1, quality=Q1, reference=R0,
                   cases=dict.fromkeys(projection.REPLAY_CASES, 'pass'), ratios=dict.fromkeys(projection.SCENES, 1.11),
                   verdict='measured', evidence_origin='operator_replay', weight_audit='pending_if_gain',
                   weighted_score=1.11, remote_report=SECRET, wall_ms=42)
        sidecar = self.root.with_name(self.root.name + '.observatory-operator.json')
        self.addCleanup(sidecar.unlink, missing_ok=True)
        doc = dict(schema='ee-observatory-operator-v1', run='example', approval_sha256=approval,
                   measurements=[dict(row, timing='b'*64, weighted_score=9), dict(row, timing=R1), row, row])
        write(sidecar, dict(doc, approval_sha256='f'*64))
        self.assertEqual([p['id'] for p in self.snapshot()['points']], [R1])
        write(sidecar, doc)
        s = self.snapshot()
        self.assertEqual([p['id'] for p in s['points']], [R1, R2])
        point = s['points'][1]
        self.assertEqual((point['index'], point['origin'], point['aggregate_kind']), (2, 'operator_replay', 'weighted_score'))
        self.assertTrue(point['provisional'])
        self.assertAlmostEqual(point['aggregate'], 1.11)
        self.assertNotIn('resources', point)
        self.assertIsNone(s['delivery'])
        self.assertEqual([e['point_id'] for e in s['events'] if e['kind'] == 'operator_measurement'], [R2])
        self.assertEqual((s['status'], s['stop_cause']), ('paused', 'operator_pause_after_segment'))

    def test_harness_recovery_is_not_load_or_success(self):
        with Journal(self.db) as db:
            db.event('level3/segment/1/start', dict(kind='initial', checkpoint=CP))
            ev(db, 'stage', 'harness_submit', version=H1, files=SECRET)
            self.assertIsNone(self.snapshot()['active_harness'])
            ev(db, 'activate', 'harness_activate', old=H0, new=H1, actor=SECRET)
            db.event('level3/segment/1/end', dict(kind='recovery', checkpoint=CP, cause='generation_truncated', next_segment=2))
            db.event('level3/segment/2/start', dict(kind='recovery', checkpoint=CP, cause='generation_truncated'))
            db.event('level3/segment/2/end', dict(kind='ordinary', checkpoint=CP, next_segment=3))
            db.event('level3/segment/3/start', dict(kind='ordinary', checkpoint=CP))
        write(self.root/'run'/'result.json', dict(segments=[dict(segment=2, report=dict(snapshot=H1, pi_tool_events=[
            dict(event='tool_execution_start', tool='read', call_sha256=CP, utc_ns=100000000, payload=SECRET),
            dict(event='tool_execution_end', tool='read', call_sha256=CP, utc_ns=107000000),
            dict(event='tool_execution_start', tool=[], call_sha256=CP, utc_ns=100000000),
        ]))]))
        s = self.snapshot()
        self.assertEqual(s['status'], 'no_terminal_receipt')
        self.assertEqual(s['active_harness'], H1)
        self.assertIsNone(s['loaded_harness'])
        self.assertEqual(next(e for e in s['events'] if e['kind'] == 'harness_activate')['previous_harness'], H0)
        ends = [e for e in s['events'] if e['kind'] == 'session_end']
        self.assertEqual([(e['continuation'], e['next_session'], e['failure']) for e in ends],
                         [('recovery', 2, True), ('ordinary', 3, False)])
        self.assertEqual(next(e for e in s['events'] if e['kind'] == 'session_harness_attached')['harness'], H1)
        self.assertEqual([e['duration_ms'] for e in s['events'] if e['kind'] == 'pi_tool'], [7])

    def test_partial_receipts_and_usage_coverage(self):
        folder = self.root/'model-segment-0001'
        write(folder/'0001-start.json', dict(started_utc='2026-11-26T03:00:00Z', messages=SECRET))
        (folder/'0001-result.json').write_bytes(b'{"usage":')
        write(folder/'0002-result.json', dict(usage=dict(usage='settled', input_tokens=10**401,
                                                         output_including_reasoning=8, finish='length')))
        write(folder/'0003-result.json', dict(usage=dict(usage='usage_uncertain')))
        s = self.snapshot()
        models = [e for e in s['events'] if e['kind'] == 'model_request']
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0]['status'], 'started')
        self.assertTrue(models[1]['failure'])
        self.assertTrue(models[2]['failure'])
        self.assertIsNone(s['resources']['input_tokens'])
        self.assertEqual(s['resources']['output_tokens'], 8)
        self.assertEqual(s['resources']['unresolved_requests'], 1)

    def test_effective_time_only_subtracts_explicit_pauses(self):
        out = projection.empty('fresh', 'FRESH', 'controller')
        out['source_ref'] = CP
        out['points'] = [{'resources': {'wall_ms': t}} for t in (50, 150, 225, 300)]
        timeline = self.root / 'nested'
        evidence = self.root / 'nested.observatory-timing.json'
        events = [dict(kind='pause', wall_ms=100, cause='worker_fault'),
                  dict(kind='resume', wall_ms=200),
                  dict(kind='infrastructure_change', wall_ms=210, infrastructure_sha256=B1),
                  dict(kind='infrastructure_fault', wall_ms=250, cause='unplanned_outage')]
        write(evidence, dict(schema='ee-observatory-timing-v1', approval_sha256=R0, events=events))
        projection.annotate_elapsed(out, timeline)
        self.assertFalse(out['events'])
        write(evidence, dict(schema='ee-observatory-timing-v1', approval_sha256=CP, events=events))
        projection.annotate_elapsed(out, timeline)
        self.assertEqual([p['resources']['effective_ms'] for p in out['points']], [50, 100, 125, 200])
        self.assertTrue(out['events'][-1]['failure'])

    def test_server_published_run_is_independent_and_missing_stays_empty(self):
        live, history = self.root/'fresh.json', self.root/'history.json'
        old = projection.empty('history', 'HISTORY', 'controller')
        old['points'] = [{'id': R0}]
        write(history, old)
        s = api.Store({}, archive=None, published={'fresh': live, 'history': history})
        self.assertEqual([r['id'] for r in s.runs()], ['fresh', 'history'])
        self.assertEqual(json.loads(s.snapshot('fresh')[1])['points'], [])
        self.assertEqual(json.loads(s.snapshot('history')[1])['points'], [{'id': R0}])
        write(live, old)  # Even a valid snapshot for another run cannot populate this curve.
        s.cache.clear()
        self.assertEqual(json.loads(s.snapshot('fresh')[1])['points'], [])

    def test_holdout_scope_is_not_projected(self):
        with Journal(self.db) as db:
            ev(db, 'registered', 'engine_registered', engine=E0)
        for extra in [dict(full_acceptance=True), dict(holdout_review_sha256=Q1), dict(scope='formal')]:
            write(self.root/'registration.json', REGISTRATION | extra)
            s = self.snapshot()
            self.assertEqual(s['status'], 'scope_not_supported')
            self.assertFalse(s['events'])
        write(self.root/'registration.json', REGISTRATION)
        write(self.root/'settlement.json', dict(holdout_released=True, status='delivered'))
        self.assertEqual(self.snapshot()['status'], 'scope_not_supported')


class _JoiningServer(api.Server):
    # Ensure all read handles have closed before deleting temporary fixtures.
    daemon_threads = False


class HttpTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name).resolve()
        self.roots = {name: base/name for name in ('one', 'two')}
        self.dbs = {name: p/'run'/'trajectory.sqlite3' for name,p in self.roots.items()}
        for name, root in self.roots.items():
            write(root/'registration.json', REGISTRATION)
            with Journal(self.dbs[name]) as db:
                ev(db, 'initial', 'engine_registered', engine=E0 if name == 'one' else E1)
        static = base/'dist'
        static.mkdir()
        (static/'index.html').write_text('test index', encoding='utf-8')
        patcher = patch.object(api, 'POLL', .03)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = _JoiningServer(('127.0.0.1', 0), api.Store(self.roots, archive=None),
                                     static=static, viewer_origin='https://rsi.example.test')
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02})
        thread.start()
        def finish():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=2)
        self.addCleanup(finish)

    def request(self, path, method='GET', **headers):
        with self.connection() as client:
            client.request(method, path, headers=headers)
            with client.getresponse() as res:
                return res.status, dict(res.headers), res.read()

    def connection(self):
        # HTTPConnection's context-manager interface is registered here explicitly.
        return _Connection('127.0.0.1', self.server.server_port, timeout=3)

    def test_http_routing_and_parallel_read_isolation(self):
        _, headers, data = self.request('/api/runs')
        self.assertEqual([x['id'] for x in json.loads(data)['runs']], ['one', 'two'])
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        for path, method, h, code in [
            ('/api/runs', 'POST', {}, 405), ('/api/runs', 'GET', {'Host': 'bad.test'}, 403),
            ('/api/runs', 'GET', {'Origin': 'https://bad.test'}, 403),
            ('/api/runs', 'GET', {'Host': 'rsi.example.test', 'Origin': 'https://rsi.example.test'}, 200),
            ('/api/runs', 'GET', {'Host': 'rsi.example.test', 'Origin': 'http://rsi.example.test'}, 403),
            ('/api/runs', 'GET', {'Sec-Fetch-Site': 'same-site'}, 403),
            ('/api/runs/one/events', 'HEAD', {}, 405),
            ('/api/runs/one/snapshot?raw=true', 'GET', {}, 404),
            ('/api/runs/one/registration.json', 'GET', {}, 404),
            ('/assets/%2e%2e/registration.json', 'GET', {}, 404),
            ('/api/runs/three/snapshot', 'GET', {}, 404),
        ]:
            with self.subTest(path=path, method=method, headers=h):
                self.assertEqual(self.request(path, method, **h)[0], code)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            one, two = pool.map(lambda n: json.loads(self.request(f'/api/runs/{n}/snapshot')[2]), ('one', 'two'))
        self.assertEqual([v['id'] for v in one['versions']], [E0])
        self.assertEqual([v['id'] for v in two['versions']], [E1])
        self.assertEqual(self.request('/')[2], b'test index')

    def test_sse_resume_and_writer_continues(self):
        path = '/api/runs/one/events'
        with Journal(self.dbs['one']) as writer:
            with self.connection() as c:
                c.request('GET', path)
                with c.getresponse() as r:
                    self.assertEqual(r.status, 200)
                    old_id, old = self.read_snapshot(r)
            with self.connection() as c:
                c.request('GET', path, headers={'Last-Event-ID': old_id})
                with c.getresponse() as r:
                    self.assertEqual(r.readline(), b'retry: 2000\n')
                    self.assertEqual(r.readline(), b'\n')
                    self.assertEqual(r.readline(), b': unchanged\n')
                    ev(writer, 'candidate', 'engine_registered', engine=E2, parent=E0)
                    new_id, new = self.read_snapshot(r)
            self.assertNotEqual(old_id, new_id)
            self.assertEqual(old['revision'], old_id)
            self.assertEqual(new['revision'], new_id)
            self.assertEqual(new['versions'][-1]['id'], E2)
            ev(writer, 'close', 'engine_registered', engine=E1, parent=E2)
            self.assertIsNotNone(writer.get_event('close'))

    def read_snapshot(self, response):
        while line := response.readline():
            if line.startswith(b'id: '):
                self.assertEqual(response.readline(), b'event: snapshot\n')
                payload = response.readline()
                self.assertTrue(payload.startswith(b'data: '))
                return line[4:].decode().strip(), json.loads(payload[6:])
        self.fail('stream closed without a snapshot')


class _Connection(http.client.HTTPConnection):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
