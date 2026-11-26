from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from evolveengine.v3_level3_evidence import (
    SCHEMA, OPERATIONS, REPORTS, CommandResult, EvidenceError, EvidencePolicy, MappedFile,
    collect_nsys_outputs, file_sha256, nsys_collection_plan, observe_loaded_mappings,
    parse_nvidia_smi_device_memory, validate_evidence,
)


QUALITY, MEASURE, PROFILE = OPERATIONS
BIND = ('schema', 'operation', 'engine', 'worker_id', 'boot_id', 'device_uuid', 'origin', 'run_id',
        'service_id', 'start_ns', 'end_ns')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(row):
    return json.dumps(row, sort_keys=True, allow_nan=False) + '\n'


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = EvidencePolicy(
            engine=sha('candidate'), worker_id='offline-worker', boot_id='offline-boot', device_uuid='fixture-device',
            weights={'weights.gguf': sha('Q8_0')}, loaded_artifacts={'engine.so': sha('build')},
            quality_rule_sha256=sha('frozen quality rule'), quality_cases=('numerics', 'natural_eos', 'cancel_mixed'),
            measure_workload_sha256=sha('frozen four scenes'), profile_workload_sha256=sha('frozen profiler scenario'),
            scenes={'W1': ((128, 8),), 'W2': ((1024, 8),), 'W3': ((4096, 8),),
                    'W4': ((128, 8), (512, 8), (1024, 8), (2048, 8))},
            warm_waves=1, timed_waves=2, memory_stages=('before', 'ready', 'post_requests'),
            memory_limit_bytes=1000, profile_target=('trusted-profile-driver', '--frozen-scenario'),
            counter_names=('DRAM Throughput', 'Occupancy'), profile_phases=('ee.decode',),
        )
        self.receipts, self.pins = {}, {}
        self.calls = []
        for i, op in enumerate(OPERATIONS):
            d = self.dir(op)
            d.mkdir()
            a = i * 1_000_000_000
            self.receipts[op] = dict(schema=SCHEMA, operation=op, engine=self.policy.engine,
                                     origin='offline_fixture', worker_id=self.policy.worker_id,
                                     boot_id=self.policy.boot_id, device_uuid=self.policy.device_uuid,
                                     run_id=f'run-{i}', service_id=f'service-{i}',
                                     observer_outside_candidate=True, profiler_enabled=op == PROFILE,
                                     start_ns=a, end_ns=a + 1_000_000_000)
        self.receipts[QUALITY].update(rule_sha256=self.policy.quality_rule_sha256, isolated_output_probe=True,
                                      memory_source='external_device_meter')
        self.receipts[MEASURE].update(workload_sha256=self.policy.measure_workload_sha256, clock='external_client_monotonic')
        self.receipts[PROFILE].update(workload_sha256=self.policy.profile_workload_sha256,
                                      hardware_counters={n: dict(status='unavailable', reason='not_requested',
                                                                source='commands.jsonl', detail='deliberately disabled')
                                                         for n in self.policy.counter_names})
        plan = nsys_collection_plan(self.policy.profile_target)
        collect_nsys_outputs(self.dir(PROFILE), plan, self.fake_nsys)
        for op in OPERATIONS:
            a = self.receipts[op]['start_ns']
            self.write(op, 'external-identity.json', dict(self.binding(op), weights_before=dict(self.policy.weights),
                       weights_after=dict(self.policy.weights), loaded_artifacts=dict(self.policy.loaded_artifacts),
                       weight_source='external hashes', loaded_source='external process mappings',
                       weights_before_at_ns=a, loaded_at_ns=a + 2, weights_after_at_ns=a + 999_999_999))
        self.quality_cases = [dict(case=name, outcome='pass', witness=name + '.json') for name in self.policy.quality_cases]
        for row in self.quality_cases:
            self.write(QUALITY, row['witness'], dict(fixture_case=row['case']))
        self.write_rows(QUALITY, 'isolated-quality.jsonl', self.quality_cases)
        self.samples = [dict(stage=name, at_ns=i + 1, used_bytes=100 * i, scope='exclusive_device', witness=f'memory-{i}.json')
                        for i, name in enumerate(self.policy.memory_stages)]
        self.write_memory()
        self.waves = []
        for scene, shape in self.policy.scenes.items():
            for i in range(3):
                a = self.receipts[MEASURE]['start_ns'] + len(self.waves) * 70_000_000
                self.waves.append(dict(scene=scene, index=i, timed=i > 0, requests=[
                    dict(input=n, output=out, rid=f'{scene}-{i}-{lane}', sha256=sha('fake tokens'),
                         start_ns=a, first_ns=a + (10_000_000 if i else 20_000_000), last_ns=a + 31_000_000,
                         done_ns=a + 40_000_000, first_token_unambiguous=True, stream_complete=True)
                    for lane, (n, out) in enumerate(shape)]))
        self.write_rows(MEASURE, 'timing.jsonl', self.waves)
        for op in OPERATIONS:
            self.seal(op)

    def dir(self, op):
        return self.root / op

    def binding(self, op):
        return {k: self.receipts[op][k] for k in BIND}

    def write(self, op, file, obj):
        (self.dir(op) / file).write_bytes(dump(obj).encode())

    def write_rows(self, op, file, rows):
        self.write(op, file, self.binding(op))
        with (self.dir(op) / file).open('ab') as f:
            f.write(''.join(map(dump, rows)).encode())

    def write_memory(self):
        self.write_rows(QUALITY, 'external-memory.jsonl', self.samples)
        for sample in self.samples:
            self.write(QUALITY, sample['witness'], dict(self.binding(QUALITY), at_ns=sample['at_ns'],
                       used_bytes=sample['used_bytes']))

    def edit_artifact(self, op, file, **edits):
        path = self.dir(op) / file
        self.write(op, file, dict(json.loads(path.read_bytes()), **edits))

    def seal(self, op):
        row = self.receipts[op]
        row['artifacts'] = {f.name: file_sha256(f) for f in self.dir(op).iterdir()
                            if f.is_file() and f.name != 'receipt.json'}
        self.write(op, 'receipt.json', row)
        self.pins[op] = file_sha256(self.dir(op) / 'receipt.json')

    def check(self):
        return validate_evidence(self.root, self.policy, self.pins)

    def operation(self, op, status, reason=''):
        row = self.check().data['operations'][op]
        self.assertEqual(row.status, status, row.reason)
        self.assertIn(reason, row.reason)
        return row

    def sql(self, commands):
        with closing(sqlite3.connect(self.dir(PROFILE) / 'trace.sqlite')) as db:
            db.executescript(commands)
            db.commit()

    def fake_nsys(self, argv, cwd):
        self.calls.append(argv)
        self.assertEqual(cwd, self.dir(PROFILE))
        if argv[1] == 'profile' and '--trace=cuda,nvtx,osrt' in argv:
            (cwd / 'trace.nsys-rep').write_bytes(b'not a real hardware capture')
            stdout = b'capture (fixture only)'
        elif 'trace.sqlite' in ' '.join(argv):
            self.sql('''
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000'), (2, 'cudaStreamSynchronize_v3020');
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId, nameId);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (5, 10, 100, 1), (12, 35, 101, 2);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start, end, correlationId);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (15, 31, 100);
                CREATE TABLE NVTX_EVENTS (start, end, text, textId);
                INSERT INTO NVTX_EVENTS VALUES (4, 36, 'ee.decode', NULL);
            ''')
            stdout = b'export (fixture only)'
        elif '--report' in argv:
            rep = argv[argv.index('--report') + 1]
            stdout = dict(cuda_gpu_trace=b'preamble\nStart (ns),Duration (ns),Name\n15,16,fixture_kernel\n',
                          cuda_gpu_kern_sum=b'Total Time (ns),Instances,Name\n16,1,fixture_kernel\n',
                          cuda_api_sum=b'Total Time (ns),Num Calls,Name\n5,1,launch\n23,1,sync\n')[rep]
        elif argv[1] == '--version':
            stdout = b'NVIDIA Nsight Systems version 2025.5.1.121'
        elif argv[1] == 'stats':
            stdout = '\n'.join(REPORTS).encode()
        elif argv[1] == 'export':
            stdout = b'--type sqlite'
        else:
            stdout = b'--trace=cuda,nvtx,osrt'
        return CommandResult(0, stdout, b'')

    def test_full_fixture_recomputes_perf_and_profile_and_marks_scope(self):
        x = self.check()
        self.assertEqual(x.status, 'valid', x.data)
        self.assertEqual(x.data['origin'], 'offline_fixture')
        self.assertTrue(x.data['profile_isolated_from_timing'])
        q, m, p = (x.data['operations'][op] for op in OPERATIONS)
        self.assertEqual((q.status, m.status, p.status), ('valid', 'observed', 'observed'))
        self.assertEqual(q.data['observed_peak_sample_bytes'], 200)
        self.assertTrue(q.data['residency_limits'])
        self.assertEqual(m.data['scenes']['W1'], dict(ttft_ms=10, tpot_ms=3, output_toks_per_s=200,dev_score=1000/3))
        self.assertEqual(m.data['scenes']['W3']['dev_score'],100)
        self.assertEqual(m.data['scenes']['W4']['dev_score'],800)
        self.assertEqual(m.data['scenes']['W4']['output_toks_per_s'], 800)
        self.assertNotIn('scenes', p.data)
        self.assertEqual(p.data['timeline']['gpu_kernel_duration_ns'], 16)
        self.assertEqual(p.data['timeline']['synchronization_api_duration_ns'], 23)
        self.assertEqual(p.data['timeline']['nvtx_phases']['ee.decode'], dict(ranges=1, total_duration_ns=32))
        self.assertEqual(p.data['hardware_counters']['Occupancy']['status'], 'unavailable')
        self.assertNotIn('samples', p.data['hardware_counters']['Occupancy'])
        self.assertFalse(p.data['counter_capability_resolved'])
        self.assertEqual(self.calls, [c.argv for c in nsys_collection_plan(self.policy.profile_target)])

    def test_unpinned_or_mutated_receipt(self):
        self.pins.pop(QUALITY)
        self.operation(QUALITY, 'insufficient_evidence', 'receipt hash')
        self.pins[QUALITY] = file_sha256(self.dir(QUALITY) / 'receipt.json')
        with (self.dir(QUALITY) / 'receipt.json').open('ab') as f:
            f.write(b' ')
        self.operation(QUALITY, 'insufficient_evidence', 'receipt hash mismatch')

    def test_artifact_hash_mismatch_is_not_candidate_violation(self):
        self.write(QUALITY, 'natural_eos.json', {'tampered': True})
        self.operation(QUALITY, 'insufficient_evidence', 'artifact hash mismatch')

    def test_path_escape_rejected_even_with_otherwise_correct_pin(self):
        row = self.receipts[QUALITY]
        row['artifacts']['../private.json'] = sha('other')
        self.write(QUALITY, 'receipt.json', row)
        self.pins[QUALITY] = file_sha256(self.dir(QUALITY) / 'receipt.json')
        self.operation(QUALITY, 'insufficient_evidence', 'unsafe archive path')

    def test_cross_engine_or_service_evidence_not_accepted_even_if_rehashed(self):
        self.edit_artifact(QUALITY, 'external-identity.json', engine=sha('other build'))
        self.seal(QUALITY)
        self.operation(QUALITY, 'insufficient_evidence', 'run binding')
        self.receipts[PROFILE]['engine'] = sha('other build')
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'identity/environment')

    def test_wrong_observed_weight_or_loaded_library_is_a_violation(self):
        for name, observed in [('weights_after', {'weights.gguf': sha('wrong weights')}),
                               ('loaded_artifacts', {'engine.so': sha('stale library')})]:
            with self.subTest(name=name):
                self.edit_artifact(QUALITY, 'external-identity.json', **{name: observed})
                self.seal(QUALITY)
                self.operation(QUALITY, 'violation', 'differs from pinned')
                self.assertEqual(self.check().status, 'violation')

    def test_real_outcome_failure_not_confused_with_missing_or_duplicate_evidence(self):
        self.quality_cases[0]['outcome'] = 'fail'
        for cases, status in [(self.quality_cases, 'violation'), (self.quality_cases[1:], 'insufficient_evidence'),
                              (self.quality_cases + self.quality_cases[:1], 'insufficient_evidence')]:
            self.write_rows(QUALITY, 'isolated-quality.jsonl', cases)
            self.seal(QUALITY)
            self.operation(QUALITY, status)

    def test_observational_memory_keeps_same_quality_and_evidence_contract(self):
        # Reproduce the READINGS of the real E0 control; these are fixture receipts.
        self.samples[0]['used_bytes'] = 5531 << 20
        self.samples[-1]['used_bytes'] = 5597 << 20
        self.write_memory(); self.seal(QUALITY)
        self.policy = replace(self.policy, memory_limit_bytes=5541 << 20)
        self.operation(QUALITY, 'violation')
        self.policy = replace(self.policy, memory_limit_bytes=None)
        row=self.operation(QUALITY, 'valid')
        self.assertEqual(row.data['weight_observation']['sampled_bytes'],
                         {s['stage']:s['used_bytes'] for s in self.samples})
        self.quality_cases[0]['outcome']='fail';self.write_rows(QUALITY,'isolated-quality.jsonl',self.quality_cases)
        self.seal(QUALITY);self.operation(QUALITY,'violation')
        self.edit_artifact(QUALITY,self.samples[-1]['witness'],used_bytes=1)
        self.seal(QUALITY);self.operation(QUALITY,'insufficient_evidence')

    def test_memory_ceiling_and_unobserved_or_wrong_window(self):
        self.samples[-1]['used_bytes'] = 1001
        self.write_memory()
        self.seal(QUALITY)
        self.operation(QUALITY, 'violation', 'observed device memory')
        self.samples[-1]['at_ns'] = self.receipts[QUALITY]['end_ns'] + 1
        self.write_memory()
        self.seal(QUALITY)
        self.operation(QUALITY, 'insufficient_evidence', 'invalid external memory sample')

    def test_external_memory_raw_witness_and_device_binding(self):
        witness = self.samples[-1]['witness']
        self.edit_artifact(QUALITY, witness, used_bytes=1200)
        self.seal(QUALITY)
        self.operation(QUALITY, 'insufficient_evidence', 'disagrees with raw witness')
        self.edit_artifact(QUALITY, witness, device_uuid='unrelated-idle-gpu')
        self.seal(QUALITY)
        self.operation(QUALITY, 'insufficient_evidence', 'run binding')

    def test_timing_and_profiling_must_not_share_service_or_window(self):
        self.receipts[PROFILE]['service_id'] = self.receipts[MEASURE]['service_id']
        self.edit_artifact(PROFILE, 'external-identity.json', service_id=self.receipts[PROFILE]['service_id'])
        self.seal(PROFILE)
        x = self.check()
        self.assertFalse(x.data['profile_isolated_from_timing'])
        self.assertEqual(x.data['operations'][MEASURE].status, 'insufficient_evidence')
        self.assertIn('reuse', x.data['operations'][PROFILE].reason)
        self.receipts[PROFILE].update(service_id='new-process', start_ns=self.receipts[MEASURE]['end_ns'] - 1)
        self.edit_artifact(PROFILE, 'external-identity.json', **self.binding(PROFILE))
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'time overlap')

    def test_profiled_timing_receipt_is_not_independent_performance(self):
        self.receipts[MEASURE]['profiler_enabled'] = True
        self.seal(MEASURE)
        self.operation(MEASURE, 'insufficient_evidence', 'instrumentation')

    def test_incomplete_or_ambiguous_timing_does_not_get_a_score(self):
        self.write_rows(MEASURE, 'timing.jsonl', self.waves[:-1])
        self.seal(MEASURE)
        self.operation(MEASURE, 'insufficient_evidence', 'incomplete timing waves')
        self.waves[-1]['requests'][0]['first_token_unambiguous'] = False
        self.write_rows(MEASURE, 'timing.jsonl', self.waves)
        self.seal(MEASURE)
        self.operation(MEASURE, 'insufficient_evidence', 'invalid timing request')

    def test_perf_summary_in_receipt_is_ignored_and_slower_timing_is_not_violation(self):
        self.receipts[MEASURE]['waves'] = [{'fake_score': 100000000}]
        self.waves[1]['requests'][0]['first_ns'] += 5_000_000
        self.waves[2]['requests'][0]['first_ns'] += 5_000_000
        self.write_rows(MEASURE, 'timing.jsonl', self.waves)
        self.seal(MEASURE)
        row = self.operation(MEASURE, 'observed')
        self.assertEqual(row.data['scenes']['W1']['ttft_ms'], 15)
        self.assertEqual(self.check().status, 'valid')

    def test_stats_capability_is_not_just_a_successful_exit_code(self):
        (self.dir(PROFILE) / 'stats_help.stdout').write_bytes(b'unrelated help')
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'report capability missing')
        (self.dir(PROFILE) / 'stats_help.stdout').write_text('\n'.join(REPORTS), encoding='utf-8')
        (self.dir(PROFILE) / 'cuda_gpu_trace.stdout').write_bytes(b'Start (ns),Duration (ns),Name\n')
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'empty Nsight GPU trace')

    def test_stats_and_export_must_agree(self):
        (self.dir(PROFILE) / 'cuda_gpu_kern_sum.stdout').write_bytes(b'Total Time (ns),Instances,Name\n32,2,kernel\n')
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'CSV/SQLite event counts or durations disagree')

    def test_sqlite_must_contain_correlated_launch_and_sync_and_bounded_phases(self):
        for command, undo, reason in [
            ('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=201;',
             'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=100;', 'correlated'),
            ('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId=2;',
             'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (12,35,101,2);', 'synchronization'),
            ('UPDATE NVTX_EVENTS SET text="untagged";', 'UPDATE NVTX_EVENTS SET text="ee.decode";', 'NVTX'),
        ]:
            self.sql(command)
            self.seal(PROFILE)
            self.operation(PROFILE, 'insufficient_evidence', reason)
            self.sql(undo)

    def test_hardware_counter_availability_requires_actual_exported_samples(self):
        self.receipts[PROFILE]['hardware_counters']['Occupancy'] = {'status': 'available'}
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'lacks exported samples')
        self.sql('''
            CREATE TABLE GPU_METRICS (timestamp, typeId, metricId, rawValue);
            CREATE TABLE TARGET_INFO_GPU_METRICS (typeId, metricId, metricName);
            INSERT INTO TARGET_INFO_GPU_METRICS VALUES (1, 0, 'Occupancy');
            INSERT INTO GPU_METRICS VALUES (15, 1, 0, 0.05), (20, 1, 0, 0.07);
        ''')
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'not enabled')
        self.policy = replace(self.policy, gpu_metrics_arg='--gpu-metrics-devices=0')
        plan = nsys_collection_plan(self.policy.profile_target, gpu_metrics_arg=self.policy.gpu_metrics_arg)
        (self.dir(PROFILE) / 'commands.jsonl').write_bytes(''.join(
            dump(dict(name=c.name, argv=c.argv, returncode=0)) for c in plan).encode())
        (self.dir(PROFILE) / 'profile_help.stdout').write_bytes(b'--trace\n--gpu-metrics-devices\n')
        self.receipts[PROFILE]['hardware_counters']['DRAM Throughput'].update(
            reason='no_samples', source='trace.sqlite', detail='No DRAM rate metric samples in this capture')
        self.seal(PROFILE)
        result = self.operation(PROFILE, 'observed').data
        self.assertEqual(result['hardware_counters']['Occupancy'], {'status': 'available', 'samples': 2})
        self.assertTrue(result['counter_capability_resolved'])

    def test_unavailable_permission_requires_archived_diagnostic(self):
        claim = dict(status='unavailable', reason='permission_denied', source='capture.stderr',
                     detail='Permission denied collecting GPU metrics')
        self.receipts[PROFILE]['hardware_counters']['Occupancy'] = claim
        self.seal(PROFILE)
        self.operation(PROFILE, 'insufficient_evidence', 'diagnostic not archived')
        (self.dir(PROFILE) / 'capture.stderr').write_text(claim['detail'], encoding='utf-8')
        self.seal(PROFILE)
        self.operation(PROFILE, 'observed')

    def test_offline_fixture_cannot_fill_hardware_policy(self):
        x = validate_evidence(self.root, replace(self.policy, origin='hardware_collection'), self.pins)
        self.assertEqual(x.status, 'insufficient_evidence')
        self.assertTrue(all('identity/environment' in c.reason for c in x.data['operations'].values()))

    def test_loaded_mapping_observer_uses_backing_file_identity(self):
        maps = ('1000-2000 r--p 0 08:01 991 /service/engine module.so\n'
                '2000-3000 r-xp 0 08:01 991 /service/engine module.so\n'
                '4000-5000 r-xp 0 00:00 0 [vdso]\n')
        paths = {'engine.so': '/service/engine module.so'}
        mapped = MappedFile(8, 1, 991, sha('loaded backing file'))
        seen = []

        def observe(addr):
            seen.append(addr)
            return mapped

        self.assertEqual(observe_loaded_mappings(maps, paths, observe), {'engine.so': mapped.sha256})
        self.assertEqual(seen, ['2000-3000'])
        with self.assertRaisesRegex(EvidenceError, 'backing file identity changed'):
            observe_loaded_mappings(maps, paths, lambda _: replace(mapped, inode=992))
        with self.assertRaisesRegex(EvidenceError, 'missing expected executable'):
            observe_loaded_mappings(maps.replace('module.so', 'module.so (deleted)'), paths, observe)
        with self.assertRaisesRegex(EvidenceError, 'multiple backing files'):
            observe_loaded_mappings(maps + '5000-6000 r-xp 0 08:01 992 /service/engine module.so\n', paths, observe)

    def test_device_memory_parser_never_converts_missing_to_zero(self):
        self.assertEqual(parse_nvidia_smi_device_memory('GPU-one, 128\nGPU-two, 0\n', 'GPU-one'), 128 << 20)
        for body in ('GPU-one, [N/A]\n', 'GPU-other, 0\n', 'GPU-one, 0\nGPU-one, 0\n'):
            with self.subTest(body=body), self.assertRaisesRegex(EvidenceError, 'unavailable or ambiguous'):
                parse_nvidia_smi_device_memory(body, 'GPU-one')

    def test_missing_cli_capability_stops_collection_before_capture(self):
        output = self.root / 'wrong-nsys'
        output.mkdir()
        plan = nsys_collection_plan(self.policy.profile_target)
        seen = []

        def run(argv, cwd):
            seen.append(argv)
            return CommandResult(0, b'Nsight Systems version 2025.5', b'')

        with self.assertRaisesRegex(EvidenceError, 'report capability missing'):
            collect_nsys_outputs(output, plan, run)
        self.assertEqual(seen, [c.argv for c in plan[:4]])
        self.assertFalse((output / 'trace.nsys-rep').exists())

    def test_collector_fails_closed_leaving_failed_command_log_and_no_later_calls(self):
        output = self.root / 'failed-probe'
        output.mkdir()
        plan = nsys_collection_plan(self.policy.profile_target)
        seen = []

        def run(argv, cwd):
            seen.append(argv)
            return CommandResult(1, b'', b'command unavailable')

        with self.assertRaisesRegex(EvidenceError, 'Nsight command failed: version'):
            collect_nsys_outputs(output, plan, run)
        self.assertEqual(seen, [plan[0].argv])
        self.assertEqual(json.loads((output / 'commands.jsonl').read_bytes())['returncode'], 1)
        self.assertEqual((output / 'version.stderr').read_bytes(), b'command unavailable')
        with self.assertRaisesRegex(EvidenceError, 'fresh profiler directory required'):
            collect_nsys_outputs(output, plan, run)


if __name__ == '__main__':
    unittest.main()
