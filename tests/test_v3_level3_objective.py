from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from evolveengine.v3_provider import LeaseError
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_worker_channel import WorkerChannelError, WorkerLease

SCRIPTS = Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf/level3'
with patch.object(sys, 'path', [str(SCRIPTS), *sys.path]):
    from pi_runtime import cfg


class LiveStreamObservationTests(TestCase):
    def test_live_stream_publishes_frame_progress_without_storing_payload(self):
        with patch.object(sys, 'path', [str(SCRIPTS), *sys.path]):
            from launch_live import observed_stream

        frames = [b': keepalive\n', b'data: first-private-delta\n', b'data: second-private-delta\n']
        with TemporaryDirectory() as directory:
            progress = Path(directory) / 'progress.json'
            times = iter(['opened', 'first', 'second'])
            source = observed_stream(frames, progress, 4, clock=lambda: 0, utc_now=lambda: next(times))
            self.assertEqual(next(source), frames[0])
            self.assertEqual(json.loads(progress.read_bytes()), dict(
                attempt=4, opened_utc='opened', last_data_utc=None, data_frames=0))
            self.assertEqual(next(source), frames[1])
            self.assertEqual(json.loads(progress.read_bytes())['data_frames'], 1)
            self.assertEqual(list(source), frames[2:])
            self.assertEqual(json.loads(progress.read_bytes()), dict(
                attempt=4, opened_utc='opened', last_data_utc='second', data_frames=2))
            self.assertEqual(list(observed_stream(frames, progress/'unwritable', 5)), frames)


class ObjectiveAdmissionTests(TestCase):
    def setUp(self):
        self.config, self.limits = cfg.validate_objective_config(json.loads(
            (SCRIPTS/'objective-exploration.json').read_bytes()))
        self.wires = []
        self.bound = 333083
        self.ledger = cfg.UsageLedger()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.req = canonical(dict(model='deepseek-flash', stream=True, thinking={'type':'enabled'},
            reasoning_effort='max', max_tokens=393216, messages=[{'role':'user','content':'test'}]))
        self.lease = cfg.ExplorationLease(self.limits, self.transport,
            SimpleNamespace(reservation=lambda req: {'local_candidate_bound':self.bound}), 1000000,
            Path(self.tmp.name)/'attempts', api_cost=self.ledger, model_socket_idle_seconds=900)
        self.lease.require_full_generation_window = False
        self.lease.minimum_generation = self.config['model']['minimum_generation_tokens']
        self.lease.context_margin = self.config['model']['context_margin_tokens']

    @contextmanager
    def transport(self, wire, timeout):
        self.wires.append(json.loads(wire))
        row = dict(model='deepseek-flash', system_fingerprint=cfg.narrow.FINGERPRINT,
                   choices=[dict(index=0,delta={},finish_reason='stop')],
                   usage=dict(prompt_tokens=400, completion_tokens=20, total_tokens=420,
                              prompt_cache_hit_tokens=350,prompt_cache_miss_tokens=50))
        yield [b'data: '+canonical(row)+b'\n', b'data: [DONE]\n']

    def test_historical_input_rejection_and_previous_caps_do_not_stop_metered_mode(self):
        self.lease.spent_input = 60000001
        self.lease.spent_output = 5000001
        self.lease.exchange(self.req)
        self.assertEqual(self.wires[0]['max_tokens'],393216)
        self.assertEqual(self.lease.spent_input,60000401)
        tally = self.ledger.snapshot()
        self.assertEqual(tally['settled_requests'],1)
        self.assertEqual(tally['unresolved_requests'],0)
        self.assertEqual(tally['estimated_rmb']['peak']['upper'],'0.000274')

    def test_near_context_limit_grants_dynamic_output_then_requests_clean_handoff(self):
        self.bound = 800000
        self.lease.exchange(self.req)
        self.assertEqual(self.wires[0]['max_tokens'],198976)
        self.lease.assistant_hashes.clear()
        self.bound = 995000
        with self.assertRaisesRegex(LeaseError,'context cannot fit'):
            self.lease.exchange(self.req)
        self.assertEqual(self.lease.rejections[-1]['reason'],'context_generation_window')
        self.assertFalse(self.lease.rejections[-1]['dispatched'])
        self.assertEqual(len(self.wires),1)

    def test_independent_worker_lease_requires_both_totals_to_be_uncapped(self):
        self.assertEqual(WorkerLease(None,None,{'develop_build':7200000}).node_ms,None)
        with self.assertRaises(WorkerChannelError):
            WorkerLease(None,10,{'develop_build':100})
        with self.assertRaises(WorkerChannelError):
            WorkerLease(100,None,{'develop_build':100})
