import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'experiments/spark26_gguf'
P=ROOT/'experiments/p2_os_boundary'
ARCH=Path(os.environ['EE_V3_ARCHIVE']) if os.getenv('EE_V3_ARCHIVE') else None

def receipt(path):return json.loads(path.read_bytes())

class PrePilotEvidence(unittest.TestCase):
    def test_8192_real_pi_receipt_has_matched_raw_outbound_clip_and_truncation(self):
        p=receipt(S/'v3_dev_rpc/launcher-max-8192-01.json')
        self.assertEqual((p['status'],p['gpu_calls'],p['paid_calls']),('pass_offline_max_output',0,0))
        self.assertEqual(p['runtime']['pi_version'],'1.5.2')
        for arm in ('B0','B2'):
            normal=p['observations']['normal'][arm]
            self.assertEqual([r['request_meta']['requested_max_tokens'] for r in normal['turns']],[8192]*3)
            self.assertEqual([r['max_tokens'] for r in normal['wire']],[8192,8192,7616])
            self.assertEqual(normal['settled_output'],24000)
            self.assertEqual(normal['pi_status'],'pi_session_complete')
            for name in ('reasoning_only','partial_tool'):
                case=p['observations'][name][arm];self.assertEqual(case['tool_events'],[]);self.assertEqual(case['trusted_events'],[])
                row,=case['turns'];self.assertEqual((row['result'],row['usage'],row['output_including_reasoning']),('generation_truncated','settled',8192))
        self.assertFalse(p['launcher_truncation']['s2_started']);self.assertFalse(p['launcher_truncation']['w']['w_started'])

    def test_independent_g1_outcomes(self):
        g=receipt(S/'g1_v/receipt-v01-20261123.json')
        self.assertEqual(g['status'],'pass_pre_pilot_lab_scope');self.assertTrue(all(g['conditions'].values()))
        self.assertEqual(g['missing_row']['status'],'insufficient_evidence')
        self.assertEqual(len(g['near_positions']),2)
        self.assertEqual(g['identity']['faultv']['qkv_events'],28*32)
        self.assertEqual(len(g['numeric_screen']['faultv']['tv_over_005']),18)
        self.assertEqual(sum(not x['exact'] for x in g['positions']['fp64greedy'].values()),27)

    def test_same_runner_freeze_isolation_and_explicit_rollback(self):
        p=receipt(P/'shared-freeze-02.json');self.assertEqual(p['status'],'pass_shared_runner_fixture');r=p['roles']
        self.assertEqual(set(r),{'train2','train0','C1','stay','revert','reopened','C2','B'})
        self.assertTrue(all(x['sandbox']['isolated_state_removed'] and x['sandbox']['socket_removed'] and x['sandbox']['unchanged_install'] for x in r.values()))
        self.assertEqual({r[k]['common_first'] for k in ['B','C1','C2']},{r['B']['common_first']})
        self.assertEqual({x['op'] for x in r['C1']['trace'] if x['kind']=='observation'},{'protected','make_private','frozen_write'})
        self.assertEqual({x['operation'] for x in r['C1']['policy'] if x['status']=='denied'},{'unregistered','harness_submit','harness_activate'})
        self.assertEqual(r['C2']['trace'][-1]['result'],{'error':'TOY_ZERO_BUG'});self.assertEqual(r['reopened']['trace'][-1]['result'],{'n':0})
        self.assertEqual(p['judge']['result']['status'],'pass')
        self.assertNotEqual(p['active_training_version'],p['frozen_version'])
        self.assertEqual(receipt(P/'receipt-03.json')['sessions']['dependency_negative']['status'],'refused_before_launch')

    def test_provider_and_dev_rpc_identity_and_cost(self):
        p=receipt(P/'live-max-01.json');self.assertEqual(p['status'],'pass_scoped');self.assertEqual([x['prior_tool_contexts'] for x in p['provider_rows']],[0,1,2])
        self.assertEqual(sum(x['output_including_reasoning'] for x in p['provider_rows']),p['spent_output'])
        self.assertTrue(all(x['effort']=='max' and x['thinking']=='enabled' for x in p['provider_rows']))
        w=receipt(S/'v3_dev_rpc/common-worker-01.json');self.assertEqual(w['status'],'pass_integrated_dev_scope')
        a,b=w['worker'];steps=w['external_policy']; self.assertEqual(a['result']['engine'],b['result']['engine']);self.assertEqual([x['status'] for x in steps],['valid','observed','selected_by_agent'])
        self.assertEqual(steps[0]['receipt_id'],steps[2]['receipt_id']);self.assertEqual(w['spent_node_ms'],sum(x['result']['node_ms'] for x in w['worker']))

    def test_launch_handoff_only_transfers_authorized_state(self):
        p=receipt(S/'v3_dev_rpc/launcher-pair-03.json');self.assertEqual(p['status'],'pass_pair_launcher_fixture')
        a,b=p['S1'],p['S2'];self.assertNotEqual(a['start_H'],b['start_H']);self.assertEqual(a['next_H'],b['start_H'])
        self.assertNotEqual(a['start_E'],b['start_E']);self.assertEqual(a['end_E'],b['start_E']);self.assertEqual(a['end_E'],b['end_E'])
        self.assertNotEqual(a['ingress_facts_sha'],b['ingress_facts_sha']);self.assertIsNone(b['explicitly_delivered'])
        self.assertEqual(p['observed']['S2:4'],'PRIVATE_ABSENT');self.assertEqual(p['observed']['S2:5'],'ACTUAL_H1_CALL')

    def test_packed_residency_is_required_even_if_numeric_quality_is_identical(self):
        p=receipt(S/'v3_dev_rpc/g1a-03.json');self.assertEqual(p['status'],'pass_g1a_pre_pilot_scope')
        e0=p['cases']['e0']['result'];bad=p['cases']['retained_qkv_bf16']['result'];q,m=p['common_runner']['worker'];q=q['result']
        self.assertEqual({e0['max_tv'],bad['max_tv'],q['max_tv']},{0});self.assertEqual({e0['eos_passed'],bad['eos_passed'],q['eos_passed']},{4})
        self.assertEqual([e0['status'],bad['status'],q['status'],m['result']['status']],['valid','violation','valid','observed'])
        self.assertEqual(bad['residency']['extra_bytes'],28*3072*2048*2);self.assertEqual(e0['residency']['packed_weights'],141)
        self.assertEqual(p['missing_phase'],'insufficient_evidence')

    @unittest.skipIf(ARCH is None,'optional machine-local raw archives')
    def test_replay_v_and_spark_worker_raw_evidence(self):
        py=lambda path,*args:subprocess.check_output([sys.executable,str(path),*map(str,args)],cwd=str(ROOT),timeout=45)
        rec=json.loads(py(S/'g1_v/audit.py',ARCH/'spark26-v32-g1-v.X6Wb1t'))
        self.assertEqual(rec['node_seconds'],445);self.assertTrue(all(rec['conditions'].values()))
        inline='from pathlib import Path; from audit_worker import audit;import json,sys;print(json.dumps(audit(*map(Path,sys.argv[1:]),legacy=True)))'
        for name,want in (('spark26-v3-rpc.vWnW3l','valid'),('spark26-v3-rpc.eF5mnP','observed')):
            b=subprocess.check_output([sys.executable,'-c',inline,str(ARCH/name),str(ARCH/'spark26-v32-g1-d3.lv2uVT'),str(ARCH/'g1-dev-fJPjW8')],
                cwd=S/'v3_dev_rpc',timeout=25)
            self.assertEqual(json.loads(b)['status'],want)

if __name__=='__main__':unittest.main()
