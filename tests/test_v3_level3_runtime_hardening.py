import io
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_guest import _cleanup, _run_pi, _wait_pi
from evolveengine.v3_level3_run import Run
from evolveengine.v3_runner import Gateway, Relay, RelayUnsettled, RpcAudit


class GatewayFaultTests(TestCase):
    def test_tool_model_and_serialization_faults_do_not_kill_next_call(self):
        tool = Mock(side_effect=[OSError('test credential never log'), (200, object()), (200, {'ok': True})])
        provider = Mock(); provider.exchange.side_effect = [AttributeError('secret provider'), b'data: fixture\n\n']
        with TemporaryDirectory() as folder:
            path = Path(folder)/'trusted.jsonl'
            gate = Gateway(provider,tool,token='fake',audit=RpcAudit(path))
            self.assertEqual(gate.dispatch('/rpc/facts',b'{}',None)[0],500)
            self.assertEqual(gate.dispatch('/rpc/facts',b'{}',None)[0],500)
            self.assertEqual(gate.dispatch('/rpc/facts',b'{}',None),(200,'application/json',b'{"ok":true}'))
            self.assertEqual(gate.dispatch('/v1/chat/completions',b'{}','Bearer fake')[0],500)
            self.assertEqual(gate.dispatch('/v1/chat/completions',b'{}','Bearer fake')[0],200)
            raw = path.read_bytes()
            for secret in (b'credential',b'secret provider',b'Bearer fake'):
                self.assertNotIn(secret,raw)
            rows = list(map(json.loads,raw.splitlines()))
            self.assertEqual([r['status'] for r in rows if r['event']=='tool_finished'],[500,500,200])
            self.assertEqual([r['route'] for r in rows if r.get('stack')],['tool','tool','provider'])
            self.assertTrue(all(r['duration_ms'] >= 0 for r in rows if r['event']=='tool_finished'))
            self.assertFalse(any('stack' in r for r in gate.events))

    def test_tool_reply_limit_and_audit_failure_do_not_break_dispatch(self):
        gate = Gateway(None,Mock(side_effect=[(200,'x'*160),(200,{})]),audit=Mock(side_effect=OSError()))
        with patch('evolveengine.v3_runner.MAX_REPLY',64):
            self.assertEqual(gate.dispatch('/rpc/facts',b'{}',None)[0],500)
            self.assertEqual(gate.dispatch('/rpc/facts',b'{}',None)[0],200)
        self.assertEqual(sum(r['kind']=='audit_unavailable' for r in gate.events),1)

    def test_same_relay_handles_a_failed_tool_then_next_frame(self):
        from base64 import b64encode
        gate = Gateway(None,Mock(side_effect=[IndexError(),(200,{'ok':True})]))
        requests = b''.join(canonical(dict(id=i,path='/rpc/facts',body=b64encode(b'{}').decode(),auth=None))+b'\n'
                            for i in (1,2))
        relay = Relay.__new__(Relay)
        relay.ready, relay.error, relay.count = Event(), None, 0
        relay.dispatch, relay.gateway = gate.dispatch, gate
        relay.frame_limit, relay.binary_replies, relay.model_reply_limit = 4096, False, 4096
        relay.proc = SimpleNamespace(stdout=io.BytesIO(b'{"ready":true}\n'+requests),stdin=io.BytesIO())
        relay._serve()
        self.assertIsNone(relay.error)
        self.assertEqual([json.loads(row)['status'] for row in relay.proc.stdin.getvalue().splitlines()],[500,200])


class WatchdogTests(TestCase):
    def relay(self, *, alive=True, error=None, started=None, worker=False):
        handler = Mock(); handler.worker_active = lambda: worker
        gate = Gateway(None,handler)
        gate.active_tool = None if started is None else (1,started)
        return SimpleNamespace(error=error, thread=SimpleNamespace(is_alive=lambda: alive),
                               proc=SimpleNamespace(poll=lambda: None),gateway=gate)

    def test_unbounded_pi_is_killed_after_relay_loss(self):
        process = Mock(); process.poll.side_effect = [None,None,None]
        process.stdout = io.BytesIO(b''); process.wait.return_value = -9
        report = {}
        with patch('evolveengine.v3_level3_guest.subprocess.Popen',return_value=process):
            _run_pi('fixture', None, report, self.relay(alive=False))
        process.kill.assert_called_once()
        self.assertEqual(report['watchdog_fault'],'relay_unavailable')
        self.assertTrue(report['pi_reader_settled'])

    def test_stale_tool_times_out_but_active_worker_and_model_do_not(self):
        with patch('evolveengine.v3_runner.time.monotonic',return_value=1000):
            for worker, start, expected in [(False,639,'tool_reply_timeout'),(True,639,None),(False,None,None),
                                             (False,641,None)]:
                process=Mock(); process.poll.side_effect=[None,None]
                report={}
                if expected is None:
                    process.poll.side_effect=[None,0]
                _wait_pi(process,None,report,self.relay(started=start,worker=worker))
                self.assertEqual(report.get('watchdog_fault'),expected)

    def test_unsettled_relay_is_reported_even_if_docker_cleanup_succeeds(self):
        report={}; relay=Mock(); relay.close.side_effect=RelayUnsettled(); relay.count=1; relay.error=None
        with patch('evolveengine.v3_level3_guest.docker',return_value=SimpleNamespace(returncode=0)), \
             patch('evolveengine.v3_level3_guest._containers_removed',return_value=True):
            _cleanup('pi','side','vol',relay,report)
        self.assertTrue(report['isolated_state_removed'] and report['socket_removed'])
        self.assertFalse(report['relay_settled'])


def state_fixture():
    state = Run.__new__(Run)
    state.scoring = True
    state.pending, state.worker_invalid, state.deadline = None, False, None
    state.rpc_unsettled = False
    state._event = Mock()
    entry = SimpleNamespace(wip=None,engine=SimpleNamespace(engine='candidate'))
    return state, entry


class BackgroundTests(TestCase):
    def test_unexpected_background_exception_is_terminal_and_next_job_can_succeed(self):
        state,entry=state_fixture()
        state._rpc=Mock(side_effect=[IndexError('secret'),(200,dict(status='ok'))])
        audited=[]; settled=Event()
        def audit(**row):
            if row['event']=='background_finished':
                audited.append(row); settled.set()
        args=dict(operation='confirm_performance',arguments={'engine':'candidate'})
        with state.handler(7,entry,audit=audit) as dispatch:
            for expected, code in [('failed',500),('settled',200)]:
                status,job=dispatch('submit_job',args)
                self.assertEqual(status,202)
                self.assertTrue(settled.wait(1)); settled.clear()
                status,row=dispatch('job_status',dict(job=job['job']))
                self.assertEqual((status,row['status'],row['http_status']),(200,expected,code))
                # Background may be finishing its final audit write; it is joined at context exit.
                from time import monotonic, sleep
                deadline=monotonic()+1
                while dispatch.worker_active() and monotonic()<deadline:
                    sleep(.001)
            self.assertFalse(dispatch.worker_active())
        self.assertEqual(audited[0]['error'],'IndexError')
        self.assertTrue(audited[0]['stack'])
        self.assertNotIn('secret',json.dumps(audited))

    def test_handler_revokes_stuck_tool_without_waiting_forever(self):
        state,entry=state_fixture(); entered=Event(); release=Event()
        def rpc(*_):
            entered.set(); release.wait(4); return 200,{}
        state._rpc=rpc
        with state.handler(7,entry) as dispatch:
            thread=Thread(target=dispatch,args=('facts',{}),daemon=True); thread.start()
            self.assertTrue(entered.wait(1))
        try:
            self.assertTrue(state.rpc_unsettled)
        finally:
            release.set(); thread.join(1)
        with self.assertRaises(ValueError):
            dispatch('facts',{})


class RecoveryIntegrationTests(TestCase):
    def exercise(self, watchdog_report, uncertain=False, stuck=False):
        from evolveengine.v3_level3_run import CORE, WorkerResult, explore
        from evolveengine.v3_level3_source import SourceContract
        from evolveengine.v3_provider import MeteredLimits

        source = SourceContract({'python/sglang/srt/engine.py':b'engine = 0\n'})
        runtime = dict(pi_image='a'*64,package_lock_sha='b'*64,runner_sha='c'*64,
                       base_config_sha='d'*64,node_version='24.13.0',pi_version='1.5.2')
        launched=[]
        entered,release=Event(),Event()
        callback=None
        original_rpc=Run._rpc
        def block(state, op, *args, **kwargs):
            if op=='facts':
                entered.set(); release.wait(8)
                return 200,{}
            return original_rpc(state,op,*args,**kwargs)
        proofs=dict(isolated_state_removed=True,socket_removed=True,unchanged_install=True,
                    relay_settled=True,pi_reader_settled=True)
        class Worker:
            verify=staticmethod(lambda _:True)
            def __call__(self, op, engine, build, cap):
                return WorkerResult(engine.digest,op,'built' if build is None else 'valid',1,
                                    None if build is None else build.evidence,
                                    ('a' if build is None else 'b')*64,{})

        def launcher(engine, snapshot, prompt, gate, **_):
            nonlocal callback
            launched.append(prompt)
            if len(launched)==1:
                if stuck:
                    callback=Thread(target=gate.dispatch,args=('/rpc/facts',b'{}',None),daemon=True)
                    callback.start()
                    self.assertTrue(entered.wait(1))
                if uncertain:
                    gate.provider.rows.append(dict(result='upstream_transport',usage='usage_uncertain'))
                return {**proofs,'watchdog_fault':'relay_unavailable','pi_exit':-1,**watchdog_report}
            self.assertEqual(gate.dispatch('/rpc/finish_exploration',canonical(dict(engine=engine.digest)),None)[0],200)
            return dict(proofs,pi_exit=0)

        with TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            def run():
                return explore(root,source,Worker(),launcher,
                    lambda *_:SimpleNamespace(rows=[],rejections=[],spent_input=0,spent_output=0),
                    runtime,'fixture',MeteredLimits(None,None,None,None,393216),node_ms=None,calls=None,
                    reservations=dict.fromkeys(CORE,100))
            if stuck:
                try:
                    with patch.object(Run,'_rpc',new=block):
                        with self.assertRaisesRegex(RuntimeError,'operator reconciliation'):
                            run()
                    self.assertFalse((root/'result.json').exists())
                    outcome=json.loads((root/'runtime-quarantine.json').read_bytes())
                finally:
                    release.set()
                    if callback:
                        callback.join(1)
            else:
                outcome=run()
        return launched,outcome

    def test_verified_cleanup_allows_recovery_from_durable_checkpoint(self):
        starts, result=self.exercise({'transport_fault':'EOFError'})
        self.assertEqual(len(starts),2)
        self.assertEqual(result['status'],'delivered')
        self.assertEqual(result['segments'][0]['entry']['engine'],result['segments'][1]['entry']['engine'])
        self.assertTrue(next(row for row in result['events'] if row['operation']=='guest_watchdog')['cleanup_verified'])

    def test_unsettled_relay_pauses_instead_of_resuming(self):
        starts, result=self.exercise({'relay_settled':False})
        self.assertEqual(len(starts),1)
        self.assertEqual(result['stop_cause'],'trusted_rpc_unsettled')

    def test_watchdog_does_not_retry_an_unsettled_model_charge(self):
        starts, result=self.exercise({},uncertain=True)
        self.assertEqual(len(starts),1)
        self.assertEqual(result['stop_cause'],'usage_uncertain')

    def test_inflight_host_callback_leaves_quarantine_not_a_new_segment(self):
        starts, result=self.exercise({'relay_settled':False},stuck=True)
        self.assertEqual(len(starts),1)
        self.assertEqual(result['stop_cause'],'trusted_rpc_unsettled')
        self.assertEqual(result['watchdog_fault'],'relay_unavailable')
