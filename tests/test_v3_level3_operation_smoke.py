from base64 import b64encode
from contextlib import closing, contextmanager
from hashlib import sha512
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_artifacts import DevelopmentArtifacts, HarnessAssets
from evolveengine.v3_level3_run import Run, WorkerResult
from evolveengine.v3_level3_scoring import NATURAL_VERSION
from evolveengine.v3_level3_web_research import WebResearch
from evolveengine.v3_runner import Gateway, RpcAudit
from experiments.spark26_gguf.level3.operation_smoke import OPERATIONS, run_cases


class OperationCoverageTests(TestCase):
    def test_positive_and_negative_cases_use_real_controller_dispatch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            npm = b'fixture registry package bytes'
            tarball = 'https://registry.npmjs.org/local-fixture/-/local-fixture-1.0.0.tgz'
            meta = canonical(dict(name='local-fixture',version='1.0.0',license='MIT',
                                  dist=dict(tarball=tarball,integrity='sha512-'+b64encode(sha512(npm).digest()).decode())))
            research = WebResearch(Mock(), root/'research')
            assets = HarnessAssets(root/'assets')
            traces = DevelopmentArtifacts()
            trace = root/'trace.sqlite'
            with closing(sqlite3.connect(trace)) as db, db:
                db.executescript('''CREATE TABLE StringIds(id,value); INSERT INTO StringIds VALUES (1,'fixture');
                                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,demangledName);
                                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100,200,1);''')
            pin = traces.publish(trace,kind='nsys_sqlite',label='fixture trace')
            cp = 'a'*64
            engine = SimpleNamespace(digest='candidate')
            journal = SimpleNamespace(get_event=lambda _:dict(checkpoint=cp))
            policy = dict(eligible_for_delivery=True,weighted_score=1.02)
            audit_path = root/'metadata.jsonl'
            audit = RpcAudit(audit_path)
            cancelled = Event()

            @contextmanager
            def open_case(name, asset):
                state = Run.__new__(Run)
                state.scoring, state.continuous, state.deadline = NATURAL_VERSION, True, None
                state.tiered_sampling, state.require_measurement = False, True
                state.worker_invalid, state.pending = False, None
                state._event = Mock()
                state.journal, state.research, state.assets = journal, research, assets
                state.next_timing_pair = 1
                chosen = SimpleNamespace(engine=engine.digest,build='build')
                state.qualified, state.engines = {engine.digest:chosen}, {engine.digest:engine}
                state.measurements = {(engine.digest,'build'):True}
                state.builds = {engine.digest:WorkerResult(engine.digest,'develop_build','built',1,None,'build',{})}
                state.worker = SimpleNamespace(artifact_query=traces.query)
                state.business_summary = lambda _: policy
                state.continuation_checkpoint = lambda _: cp
                state.lifecycle = SimpleNamespace(load=lambda _:SimpleNamespace(harness='active'))
                state.lineage = SimpleNamespace(active=lambda:SimpleNamespace(identity='active'),catalogue=lambda:[])
                state.develop = Mock(return_value=(dict(status='observed') if name=='develop_stress' else
                                                 dict(status='built',data={'build_mode':'independent_cold'})))
                unblock = Event()
                def confirm(_engine, signal=None):
                    if signal is not None:
                        if signal.wait(1):
                            cancelled.set()
                        unblock.wait(2)
                    return dict(status='eligible')
                state.confirm_performance = confirm
                entry = SimpleNamespace(wip=None,engine=chosen)
                with state.handler(1,entry,audit=audit) as dispatch:
                    gate = Gateway(None,dispatch,audit=audit)
                    args = dict(engine=engine.digest)
                    if name == 'dependency_fetch':
                        args = dict(name='local-fixture',version='1.0.0')
                    elif name == 'research_reuse':
                        args = dict(kind='original',paths=['fixture.py'])
                    elif name == 'artifact_query':
                        args = dict(view='kernels',identity=pin,limit=1)
                    elif name == 'continue_session':
                        args = dict(checkpoint=cp)
                    elif name == 'job_cancel':
                        code,_,raw = gate.dispatch('/rpc/submit_job',canonical(dict(operation='confirm_performance',
                                                      arguments=args)),None)
                        self.assertEqual(code,202)
                        args = dict(job=json.loads(raw)['job'])
                    try:
                        yield gate,args
                        if name == 'job_cancel':
                            self.assertTrue(cancelled.wait(1))
                        if name == 'develop_stress':
                            state.develop.assert_called_once_with('develop_measure',engine,timing_pair=1,stress=True)
                        if name == 'confirm_cold_build':
                            state.develop.assert_called_once_with('develop_build',engine,independent=True)
                    finally:
                        unblock.set()

            def registry(url, _limit):
                return npm if url==tarball else meta
            with patch('evolveengine.v3_level3_dependencies._get',side_effect=registry):
                results = run_cases(open_case)
            self.assertEqual([row['tool'] for row in results],list(OPERATIONS))
            rows = [json.loads(line) for line in audit_path.read_bytes().splitlines()]
            for name in OPERATIONS:
                route = {'harness_asset_save':'harness_asset_put','harness_asset_restore':'harness_asset_read'}.get(name,name)
                codes = [r['status'] for r in rows if r['event']=='tool_finished' and r['tool']==route]
                self.assertTrue(any(c==200 for c in codes), (name,codes))
                self.assertTrue(any(400<=c<500 for c in codes), (name,codes))
            self.assertTrue(any(row.get('event')=='background_finished' and row.get('status')==200 for row in rows))
