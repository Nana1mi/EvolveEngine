from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from experiments.spark26_gguf.level3 import profile_control as r


class ProfileControlTests(TestCase):
    def test_named_gpu_event_and_not_just_string_table(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript('''CREATE TABLE StringIds(id, value);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start, end, demangledName);
                    INSERT INTO StringIds VALUES (1,'another_kernel'), (2,'void ee_level3_positive_mmvq_control<T>()');
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7,10,1);''')
            symbol = r.native_controls.KERNEL_WITNESS
            with self.assertRaisesRegex(ValueError,'did not execute'):
                r.kernel_execution(trace,symbol)
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (11, 19, 2), (20, 28, 2)')
            self.assertEqual(r.kernel_execution(trace,symbol),dict(symbol=symbol,launches=2,device_duration_ns=16))
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (31, 21, 2)')
            with self.assertRaisesRegex(ValueError,'interval'):
                r.kernel_execution(trace,symbol)

    def test_software_trace_allows_zero_ns_host_api_but_not_zero_ns_kernels(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript('''CREATE TABLE StringIds(id,value);
                    INSERT INTO StringIds VALUES (1,'cudaLaunchKernel_v7000'),(2,'cudaStreamSynchronize_v3020');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,correlationId,nameId);
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2,2,1,1),(6,8,2,2);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,correlationId);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (3,5,1);''')
            with self.assertRaisesRegex(ValueError, 'relative trace interval'):
                r._sqlite_trace(trace, 10, ())
            timeline, _, _, _ = r._sqlite_trace(trace, 10, (), allow_zero_api_duration=True)
            self.assertEqual(timeline['zero_duration_api_calls'], 1)
            self.assertEqual(timeline['launch_api_duration_ns'], 0)
            self.assertEqual(timeline['correlated_kernels'], 1)
            self.assertEqual(timeline['synchronization_api_calls'], 1)
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start')
            with self.assertRaisesRegex(ValueError, 'relative trace interval'):
                r._sqlite_trace(trace, 10, (), allow_zero_api_duration=True)

    def test_plan_pins_fixture_mode_and_excludes_scoring(self):
        e0 = r.make_plan('0'*64,{'build':'/path'},'e0')
        pos = r.make_plan('0'*64,{'plan':'/path'},'native_positive')
        self.assertEqual(e0['expected_kernel'],'mul_mat_vec_q')
        self.assertEqual(pos['expected_kernel'],r.native_controls.KERNEL_WITNESS)
        self.assertFalse(pos['scoring'])
        self.assertFalse(pos['holdout'])
        self.assertIn('--export=sqlite',pos['profile_argv'])
        self.assertIn('--sample=none',pos['profile_argv'])
        self.assertEqual(e0['report_storage'], 'stream_to_host_with_disk_reserve')
        self.assertEqual(pos['report_storage'], e0['report_storage'])
        graph_reg = r.make_plan('0'*64, {'plan':'/path'}, 'native_positive', 'graph_defaults')
        self.assertEqual(graph_reg['profile_argv'], pos['profile_argv'])
        self.assertIn('--trace=cuda,cuda-sw,nvtx', graph_reg['profile_argv'])
        self.assertIn('--kill=none', graph_reg['profile_argv'])
        self.assertIn('--disable-decode-cuda-graph', graph_reg['profile_argv'])
        self.assertIn('--disable-prefill-cuda-graph', graph_reg['profile_argv'])

    def test_graph_mode_pins_runtime_flags_and_requires_separate_profile(self):
        with self.assertRaisesRegex(ValueError, 'Graph profile'):
            r.make_plan('0'*64, {}, 'e0_graph')
        decode = r.make_plan('0'*64, {}, 'e0_graph', 'decode_graph')
        self.assertIn('--cuda-graph-trace=graph:host-only', decode['profile_argv'])
        self.assertIn('--trace=cuda,cuda-sw,nvtx', decode['profile_argv'])
        self.assertIn('--disable-prefill-cuda-graph', decode['profile_argv'])
        self.assertNotIn('--disable-decode-cuda-graph', decode['profile_argv'])
        both = r.make_plan('0'*64, {}, 'e0_graph', 'graph_defaults')
        self.assertNotIn('--disable-prefill-cuda-graph', both['profile_argv'])
        self.assertFalse(both['scoring'])

    def test_incremental_graph_profile_uses_the_registered_service_args(self):
        from types import SimpleNamespace
        from evolveengine.v3_level3_deploy import IMMEDIATE_GRAPH_WIRE, launch_args

        spec = SimpleNamespace(launch_profile=IMMEDIATE_GRAPH_WIRE)
        command = r.trace_args(spec.launch_profile, 'e0_graph')
        self.assertIn('--cuda-graph-trace=graph:host-only', command)
        self.assertEqual(command[command.index('--')+2:], tuple(launch_args(spec)))
        self.assertIn('--incremental-streaming-output', command)

    def test_graph_trace_admits_new_graph_profiles_from_the_shared_registry(self):
        from unittest.mock import patch
        from evolveengine.v3_level3_deploy import GRAPH_PROFILES

        with patch.dict(GRAPH_PROFILES, {'fixture_graph': frozenset({'--disable-decode-cuda-graph'})}):
            self.assertNotIn('--disable-decode-cuda-graph',r.trace_args('fixture_graph','e0_graph'))
        for profile in ('diagnostic_eager','unknown'):
            with self.assertRaisesRegex(ValueError,'registered Graph'):
                r.trace_args(profile,'e0_graph')

    def test_graph_rejects_startup_and_warm_activity(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            unit = 1_000_000
            offset = 1000*unit
            launch = (45*unit, 50*unit, 0, 5)
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript(f"""CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
                    INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES ({offset});
                    CREATE TABLE {r.GRAPH_TRACE}(start,end,deviceId,graphId);""")
                # Warm, outside the timed window, host, missing Graph ID, and no device timing.
                conn.executemany(f'INSERT INTO {r.GRAPH_TRACE} VALUES (?, ?, ?, ?)',
                                 [(5*unit, 8*unit, 0, 3), (30*unit, 34*unit, 0, 4),
                                  (45*unit, 50*unit, 2**31-1, 5), (45*unit, 50*unit, 0, 0), (0, 0, 0, 5)])
            rows = [dict(scene='W1',timed=False,requests=[dict(first_ns=0,last_ns=20*unit)]),
                    dict(scene='W1',timed=True,requests=[dict(first_ns=30*unit,last_ns=90*unit)])]
            clock = (offset, offset+1)
            self.assertFalse(r.graph_execution(trace,rows,clock,clock)['observed'])
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.execute(f'INSERT INTO {r.GRAPH_TRACE} VALUES (?, ?, ?, ?)', launch)
            self.assertEqual(r.graph_execution(trace,rows,clock,clock)['scenes'],
                             {'W1': dict(graph_executions=1,graph_duration_ns=5*unit,graph_node_executions=0,graph_node_duration_ns=0,
                                  graph_api_calls=0,graph_api_successes=0,graph_api_failures=0,graph_api_duration_ns=0)})
            with self.assertRaisesRegex(ValueError, 'clock alignment'):
                r.graph_execution(trace,rows,clock,(offset+30*unit,offset+31*unit))

    def test_graph_counts_whole_graph_and_kernel_nodes_separately_when_both_tables_exist(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            unit = 1_000_000
            offset = 1000*unit
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript(f"""CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
                    INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES ({offset});
                    CREATE TABLE {r.GRAPH_TRACE}(start,end,deviceId,graphId);
                    INSERT INTO {r.GRAPH_TRACE} VALUES ({45*unit},{50*unit},0,7);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,graphId,graphNodeId);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({46*unit},{49*unit},0,7,9);""")
            rows = [dict(scene='W1',timed=True,requests=[dict(first_ns=30*unit,last_ns=90*unit)])]
            clock = (offset, offset+1)
            result = r.graph_execution(trace,rows,clock,clock)
            self.assertEqual(result['schema'], 'level3_cuda_graph_observation_v2')
            self.assertIn('nsight_graph_node_execution_during_timed_decode_v1', result['evidence_sources'])
            self.assertEqual(result['scenes']['W1'], dict(graph_executions=1,graph_duration_ns=5*unit,
                                                            graph_node_executions=1,graph_node_duration_ns=3*unit,
                                                            graph_api_calls=0,graph_api_successes=0,graph_api_failures=0,graph_api_duration_ns=0))

    def test_successful_graph_launch_runtime_api_is_scoped_to_timed_real_text(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            unit = 1_000_000
            offset = 1000*unit
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript(f"""CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
                    INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES ({offset});
                    CREATE TABLE StringIds(id,value);
                    INSERT INTO StringIds VALUES (1,'cudaGraphLaunch_v10000'),(2,'cudaGraphDestroy_v10000');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,returnValue,nameId);
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                      ({45*unit},{46*unit},0,1),({47*unit},{48*unit},719,1),({49*unit},{50*unit},0,2);""")
            rows = [dict(scene='W1',timed=True,requests=[dict(first_ns=30*unit,last_ns=90*unit)])]
            clock = (offset, offset+1)
            result = r.graph_execution(trace,rows,clock,clock)
            self.assertFalse(result['observed'])
            self.assertTrue(result['host_submission_observed'])
            self.assertIn('successful_cuda_graph_launch_runtime_api_during_timed_decode_v1', result['evidence_sources'])
            self.assertIn('not per-node device completion', result['api_launch_semantics'])
            self.assertEqual(result['scenes']['W1'], dict(graph_executions=0,graph_duration_ns=0,
                graph_node_executions=0,graph_node_duration_ns=0,graph_api_calls=2,
                graph_api_successes=1,graph_api_failures=1,graph_api_duration_ns=2*unit))

    def test_graph_falls_back_to_kernel_graph_ids_without_claiming_whole_graph(self):
        with TemporaryDirectory() as folder:
            trace = Path(folder).resolve()/'trace.sqlite'
            unit = 1_000_000
            offset = 1000*unit
            with closing(sqlite3.connect(trace)) as conn, conn:
                conn.executescript(f"""CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
                    INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES ({offset});
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,graphId,graphNodeId);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({45*unit},{50*unit},0,7,9);""")
            rows = [dict(scene='W1',timed=True,requests=[dict(first_ns=30*unit,last_ns=90*unit)])]
            clock = (offset, offset+1)
            result = r.graph_execution(trace,rows,clock,clock)
            self.assertEqual(result['schema'], 'level3_cuda_graph_observation_v2')
            self.assertIn('nsight_graph_node_execution_during_timed_decode_v1', result['evidence_sources'])
            self.assertTrue(result['observed'])
            self.assertEqual(result['scenes']['W1'], dict(graph_executions=0,graph_duration_ns=0,
                                                            graph_node_executions=1,graph_node_duration_ns=5*unit,
                                                            graph_api_calls=0,graph_api_successes=0,graph_api_failures=0,graph_api_duration_ns=0))
