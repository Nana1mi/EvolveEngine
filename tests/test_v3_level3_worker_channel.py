from shlex import split
from socket import socketpair
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from base64 import b64decode
from contextlib import closing
from io import BytesIO
import sqlite3
from threading import Event, Thread
from unittest import TestCase
from unittest.mock import patch

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_artifacts import DevelopmentArtifacts, PAGE_BYTES
from evolveengine.v3_level3_run import OPS, WorkerResult
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_level3_worker_channel import (
    REPLY, REQUEST, WorkerChannelError, WorkerClient, WorkerLease, _binding, _get, _send, serve,
)
from evolveengine.v3_level3_worker_spool import SpoolIO, bridge, spool_path
from evolveengine.v3_level3_worker_ssh import _worker_command


class Worker:
    def __init__(self):
        self.calls = []
        self.records = {}

    def __call__(self, op, candidate, build, cap):
        address = digest(canonical([len(self.calls),op,candidate.digest]))
        result = WorkerResult(candidate.digest,op,'built' if op=='develop_build' else 'valid',9,
                              None if op=='develop_build' else build.evidence,address,{'fixture':True})
        self.calls.append((op,build))
        self.records[address] = result
        return result

    def measure(self, candidate, build, cap, pair_index):
        address = digest(canonical(['paired timing', candidate.digest, build.evidence, pair_index]))
        result = WorkerResult(candidate.digest, 'develop_measure', 'observed', 9, build.evidence,
                              address, {'measurement_pair': pair_index})
        self.calls.append(('develop_measure', build))
        self.records[address] = result
        return result

    def independent_build(self, candidate, cap):
        result = self('develop_build',candidate,None,cap)
        result = replace(result,data={'build_mode':'independent_cold'})
        self.records[result.evidence] = result
        return result

    def diagnose(self, candidate, build, source, cap):
        address = digest(canonical(['diagnostic',candidate.digest,build.evidence,source]))
        result = WorkerResult(candidate.digest,'develop_diagnostic','completed_non_scoring',9,build.evidence,
                             address,{'script_sha256':digest(source.encode()),'non_scoring':True})
        self.records[address] = result
        return result

    def verify(self, result):
        return self.records.get(result.evidence)==result


class ChannelTests(TestCase):
    def test_spool_replays_the_receipt_after_bridge_disconnect_without_reexecution(self):
        with TemporaryDirectory() as folder:
            report = Path(folder)/'report'
            report.mkdir()
            spool = spool_path(report)
            spool.mkdir()
            pin = 'b'*64
            (spool/'authorization').write_text(pin)
            contract = SourceContract({'python/sglang/test.py':b'base\n'})
            worker = Worker()
            engine = contract.from_args({'files':{}})
            lease = WorkerLease(None,None,{'develop_build':50})
            with patch('evolveengine.v3_level3_worker_spool._start_ticks',return_value=1):
                stream = SpoolIO(report,pin)
            thread = Thread(target=serve,args=(stream,stream,contract,worker,pin,lease),daemon=True)
            thread.start()
            request = dict(kind='run',seq=1,registration=pin,operation='develop_build',engine=engine.digest,
                           candidate=contract.to_args(engine),parent_build=None,cap_ms=50)
            first_in,first_out = BytesIO(),BytesIO()
            _send(first_in,request,REQUEST)
            first_in.seek(0)
            with patch('evolveengine.v3_level3_worker_spool.os.getuid',return_value=spool.stat().st_uid,create=True):
                with self.assertRaisesRegex(WorkerChannelError,'channel closed'):
                    bridge(report,pin,[],first_in,first_out)
                first_out.seek(0)
                self.assertEqual(_get(first_out,REPLY)['kind'],'hello')
                prior = _get(first_out,REPLY)
                again_in,again_out = BytesIO(),BytesIO()
                _send(again_in,request,REQUEST)
                _send(again_in,dict(kind='close',seq=2,registration=pin),REQUEST)
                again_in.seek(0)
                bridge(report,pin,[],again_in,again_out)
                again_out.seek(0)
                self.assertEqual(_get(again_out,REPLY)['kind'],'hello')
                self.assertEqual(_get(again_out,REPLY),prior)
                self.assertEqual(_get(again_out,REPLY)['kind'],'closed')
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(worker.calls),1)
    def test_large_artifacts_are_paged_without_a_worker_job_or_raw_host_path(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            output = root/'observer.stdout'
            raw = b'0123456789abcdef' * 20000
            output.write_bytes(raw)
            trace = root/'trace.sqlite'
            with closing(sqlite3.connect(trace)) as db, db:
                db.executescript('CREATE TABLE StringIds(id, value);'
                                 'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,demangledName);'
                                 "INSERT INTO StringIds VALUES(1,'decode'),(2,'prefill');"
                                 'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,9,1),(20,29,1),(30,40,2);')
            catalog = DevelopmentArtifacts()
            pin = catalog.publish(output,kind='diagnostic_stdout',label='stdout',expected=digest(raw))
            sql = catalog.publish(trace,kind='nsys_sqlite',label='trace')
            c,cr,cw,contract,worker,reg,t,errors = self.setup_worker()
            worker.artifact_query = catalog.query
            client = WorkerClient.from_channel(c,contract,reg)
            try:
                self.assertEqual(len(client.artifact_query('list')['items']),2)
                runs = client.artifact_query('kernels',sql,limit=1)
                self.assertEqual(runs['kernels'],[dict(name='decode',calls=2,device_ns=17)])
                self.assertEqual(client.artifact_query('kernels',sql,offset=runs['next_offset'],limit=1)
                                 ['kernels'][0]['name'],'prefill')
                recovered = bytearray()
                while len(recovered)<len(raw):
                    page = client.artifact_query('read',pin,offset=len(recovered))
                    piece = b64decode(page['data_b64'],validate=True)
                    self.assertLessEqual(len(piece), PAGE_BYTES)
                    self.assertEqual(digest(piece),page['chunk_sha256'])
                    recovered.extend(piece)
                self.assertIsNone(page['next_offset'])
                self.assertEqual(digest(recovered),pin)
                with self.assertRaises(ValueError):
                    client.artifact_query('read',str(output))
                output.write_bytes(raw+b'x')
                with self.assertRaises(ValueError):
                    client.artifact_query('read',pin)
                self.assertEqual(len(client.artifact_query('list')['items']),2)
            finally:
                client.close()
                self.ack_received.set()
                t.join(5)
            self.assertEqual(errors,[])
            self.assertEqual(worker.calls,[])

    def test_remote_stdio_command_has_no_agent_shell_arguments(self):
        opts = dict(entry='/var/tmp/trusted/experiments/spark26_gguf/level3/worker_stdio.py',
                    judge_deps='/var/tmp/deps',report='/var/tmp/report',registration='/var/tmp/reg.json',
                    authorization='/var/tmp/auth.json',pin='a'*64)
        command = split(_worker_command(**opts))
        self.assertEqual(command[:7],['/usr/bin/sudo','-k','-S','-p','','--','/usr/bin/env'])
        self.assertEqual(command[-10:],['/usr/bin/python3','-Bu',opts['entry'],opts['report'],'--registration',
                                        opts['registration'],'--authorization',opts['authorization'],'--sha256',opts['pin']])
        for illegal in ('/var/tmp/x;whoami','/var/tmp/../etc/passwd','/home/tier/x'):
            with self.subTest(path=illegal),self.assertRaises(ValueError):
                _worker_command(**dict(opts,report=illegal))

    def test_engineering_lease_can_omit_unqualified_lanes(self):
        lease = WorkerLease(100,1,{'develop_build':50})
        self.assertNotIn('develop_profile',lease.max_per_op)
        with self.assertRaises(WorkerChannelError):
            WorkerLease(100,1,{'holdout':50})

    def setup_worker(self, *, jobs=2, node_ms=100, max_per_op=50, e0_only=False):
        c,s = socketpair()
        c.settimeout(5)
        s.settimeout(5)
        cr,cw = c.makefile('rb',buffering=0),c.makefile('wb',buffering=0)
        self.addCleanup(cr.close)
        self.addCleanup(cw.close)
        self.addCleanup(c.close)
        contract = SourceContract({'python/sglang/test.py':b'base\n'})
        worker = Worker()
        pin = 'a'*64
        errors = []
        self.ack_received = Event()
        def run():
            with s,s.makefile('rb',buffering=0) as fr,s.makefile('wb',buffering=0) as fw:
                try:
                    serve(fr,fw,contract,worker,pin,WorkerLease(node_ms,jobs,dict.fromkeys(OPS,max_per_op)),e0_only=e0_only)
                except WorkerChannelError as e:
                    errors.append(str(e))
                else:
                    # Keep the test transport open until the client consumed
                    # the final frame; Windows socketpair may reset an unread ACK.
                    if not self.ack_received.wait(5):
                        errors.append('test client did not read the close frame')
        t = Thread(target=run,daemon=True)
        t.start()
        return c,cr,cw,contract,worker,pin,t,errors

    def test_e0_only_rehearsal_cannot_admit_candidate(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker(e0_only=True)
        client = WorkerClient.from_channel(c,contract,pin)
        engine = contract.from_args({'files':{'python/sglang/test.py':'other\n'}})
        with self.assertRaises(WorkerChannelError):
            client('develop_build',engine,None,50)
        t.join(5)
        self.assertEqual(worker.calls,[])
        self.assertEqual(errors,['engineering Worker only accepts E0'])

    def test_build_quality_and_reverify_bind_both_hosts(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        client = WorkerClient.from_channel(c,contract,pin)
        engine = contract.from_args({'files':{'python/sglang/test.py':'changed\n'}})
        build = client('develop_build',engine,None,50)
        self.assertTrue(client.verify(build))
        result = client('develop_quality',engine,build,50)
        self.assertTrue(client.verify(result))
        self.assertEqual(result.build,build.evidence)
        client.close()
        self.ack_received.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors,[])
        self.assertEqual(len(worker.calls),2)

    def test_distinct_candidate_can_reuse_a_verified_build_only_for_building(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker(jobs=3, node_ms=150)
        client = WorkerClient.from_channel(c,contract,pin)
        base = contract.from_args({'files':{}})
        candidate = contract.from_args({'files':{'python/sglang/test.py':'new child'}})
        prior = client('develop_build',base,None,50)
        child = client('develop_build',candidate,prior,50)
        self.assertTrue(client.verify(child))
        self.assertEqual(worker.calls[-1],('develop_build',prior))
        with self.assertRaises(WorkerChannelError):
            client('develop_quality',candidate,prior,50)
        t.join(5)
        self.assertEqual(len(worker.calls),2)
        self.assertEqual(errors,['parent does not identify a verified Worker build'])

    def test_cold_verification_has_no_parent_and_is_a_distinct_protocol_operation(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        client = WorkerClient.from_channel(c,contract,pin)
        candidate = contract.from_args({'files':{'python/sglang/test.py':'candidate'}})
        result = client.independent_build(candidate,50)
        self.assertTrue(client.verify(result))
        self.assertEqual(result.data,{'build_mode':'independent_cold'})
        self.assertIsNone(result.build)
        client.close()
        self.ack_received.set()
        t.join(5)
        self.assertEqual(errors,[])

    def test_bounded_diagnostic_is_a_distinct_non_scoring_worker_action(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        client = WorkerClient.from_channel(c,contract,pin)
        engine = contract.from_args({'files':{'python/sglang/test.py':'changed\n'}})
        build = client('develop_build',engine,None,50)
        source = "print('probe')\n"
        result = client.diagnose(engine,build,source,50)
        self.assertEqual(result.operation,'develop_diagnostic')
        self.assertEqual(result.status,'completed_non_scoring')
        self.assertEqual(result.build,build.evidence)
        self.assertTrue(client.verify(result))
        client.close()
        self.ack_received.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors,[])

    def test_paired_timing_survives_worker_wire_and_evidence_verification(self):
        channel, _, _, contract, worker, pin, server, errors = self.setup_worker()
        client = WorkerClient.from_channel(channel, contract, pin)
        engine = contract.from_args({'files': {}})
        build = client('develop_build', engine, None, 50)
        observed = client.measure(engine, build, 50, 17)
        self.assertEqual((observed.status, observed.build), ('observed', build.evidence))
        self.assertEqual(observed.data['measurement_pair'], 17)
        self.assertTrue(client.verify(observed))
        client.close()
        self.ack_received.set()
        server.join(5)
        self.assertFalse(server.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([op for op, _ in worker.calls], ['develop_build', 'develop_measure'])

    def test_sampling_draw_charged_as_quality_without_granting_quality_acceptance(self):
        channel, _, _, contract, worker, pin, server, errors = self.setup_worker()
        client = WorkerClient.from_channel(channel,contract,pin)
        engine=contract.from_args({'files':{}}); b=client('develop_build',engine,None,50)
        def draw(e,parent,cap,spec):
            row=worker('develop_quality',e,parent,cap)
            row=replace(row,status='sampled',data={'challenge':spec,'decision':'reference'})
            worker.records[row.evidence]=row;return row
        worker.sampling=draw
        spec=dict(milestone='final_nomination',stage=1,role='reference')
        row=client.sampling(engine,b,50,spec)
        self.assertEqual((row.status,row.build,row.node_ms,row.data['challenge']),('sampled',b.evidence,9,spec))
        self.assertTrue(client.verify(row));client.close();self.ack_received.set();server.join(5)
        self.assertFalse(server.is_alive());self.assertEqual(errors,[])

    def test_refusing_budget_is_terminal_and_no_retry_is_dispatched(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker(jobs=1)
        client = WorkerClient(cr,cw,contract,pin,set_timeout=lambda _:None)
        engine = contract.from_args({'files':{}})
        build = client('develop_build',engine,None,50)
        for _ in range(2):
            with self.assertRaises(WorkerChannelError):
                client('develop_quality',engine,build,50)
        t.join(5)
        self.assertEqual(len(worker.calls),1)
        self.assertEqual(errors,['Worker authorization exceeded'])

    def test_worker_replays_the_exact_candidate_and_refuses_forged_parent(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        client = WorkerClient(cr,cw,contract,pin,set_timeout=lambda _:None)
        engine = contract.from_args({'files':{}})
        build = client('develop_build',engine,None,50)
        with self.assertRaises(WorkerChannelError):
            client('develop_quality',engine,replace(build,evidence='b'*64),50)
        t.join(5)
        self.assertEqual(errors,['parent does not identify a verified Worker build'])
        self.assertEqual(len(worker.calls),1)

    def test_verification_rechecks_storage_not_just_shape(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        client = WorkerClient(cr,cw,contract,pin,set_timeout=lambda _:None)
        engine = contract.from_args({'files':{}})
        build = client('develop_build',engine,None,50)
        worker.records.pop(build.evidence)
        self.assertFalse(client.verify(build))
        with self.assertRaises(WorkerChannelError):
            client('develop_quality',engine,build,50)
        client.close()
        self.ack_received.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors,[])
        self.assertEqual(len(worker.calls),1)

    def test_interrupt_reaches_waiting_worker_and_rejects_its_late_reply(self):
        contract = SourceContract({'python/sglang/test.py': b'base\n'})
        registration = 'a' * 64
        engine = contract.from_args({'files': {}})
        waiting, interrupted = Event(), Event()
        outcomes = []

        def late_reply():
            waiting.set()
            if not interrupted.wait(3):
                raise AssertionError('Interrupt did not reach the busy Worker')
            return dict(seq=1, kind='result')

        def dispatch():
            try:
                outcomes.append(client('develop_build', engine, None, 50))
            except WorkerChannelError as exc:
                outcomes.append(exc)

        with patch('evolveengine.v3_level3_worker_channel._send') as send, \
                patch('evolveengine.v3_level3_worker_channel._get') as receive:
            receive.side_effect = lambda *args: late_reply() if send.called else dict(
                kind='hello', **_binding(contract, registration))
            client = WorkerClient(None, None, contract, registration, set_timeout=lambda _: None,
                                  interrupt_transport=interrupted.set)
            request = Thread(target=dispatch, daemon=True)
            request.start()
            self.assertTrue(waiting.wait(3))
            client.interrupt()
            client.interrupt()
            request.join(3)
            self.assertFalse(request.is_alive())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], WorkerChannelError)
            self.assertIn('reply arrived after interruption', str(outcomes[0].__cause__))
            with self.assertRaisesRegex(WorkerChannelError, 'not open'):
                client('develop_build', engine, None, 50)
            self.assertEqual(send.call_count, 1)

    def test_wrong_worker_pin_refuses_session(self):
        c,cr,cw,contract,worker,pin,t,errors = self.setup_worker()
        with self.assertRaisesRegex(WorkerChannelError,'does not match'):
            WorkerClient(cr,cw,contract,'b'*64,set_timeout=lambda _:None)
        c.shutdown(2)
        t.join(5)
        self.assertEqual(worker.calls,[])
