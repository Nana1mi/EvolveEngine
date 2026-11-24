from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import runtime_identity as identity


ARGS = {'load_format':'gguf', 'attention_backend':'triton', 'prefill_attention_backend':None,
        'decode_attention_backend':None, 'kv_cache_dtype':'auto', 'tp_size':1,
        'enable_deterministic_inference':False, 'mem_fraction_static':0.4,
        'api_key':'do-not-export-me', 'model_path':'/very/private/path', 'host':'some.instance'}
START = '[2026-11-23 06:54:31] server_args='
CAP = ('[2026-11-23 06:54:39 PID113] max_total_num_tokens=4096, chunked_prefill_size=8192, '
       'max_prefill_tokens=16384, max_running_requests=2, context_len=1024, available_gpu_mem=107.48 GB')
KV = ('[2026-11-23 06:54:39 PID113] Full KV Cache is allocated. dtype: torch.bfloat16, '
      '#tokens: 4096, K size: 0.03 GB, V size: 0.03 GB')


def log(args=ARGS, *more, capacity=CAP, pool=KV):
    return ('\n'.join([START+repr(args), capacity, pool,*more])+'\n').encode()


class RuntimeIdentityTests(unittest.TestCase):
    def test_redacts_noncontract_and_separates_requested_from_effective(self):
        r=identity.project(log())
        self.assertEqual(r['startup_args']['prefill_attention_backend'],None)
        self.assertEqual(r['startup_args']['kv_cache_dtype'],'auto')
        self.assertEqual(r['scheduler_capacity']['max_total_num_tokens'],4096)
        self.assertEqual(r['kv_pools']['full'],{'dtype':'bfloat16','tokens':4096})
        self.assertEqual(r['unavailable_startup_args']['context_length'],'not_present_in_log')
        packed=json.dumps(r)
        for private in ('api_key','do-not-export-me','model_path','/very/private/path','some.instance'):
            self.assertNotIn(private,packed)

    def test_flags_can_match_while_scheduler_capacity_differs(self):
        a=identity.project(log())
        b=identity.project(log(capacity=CAP.replace('max_total_num_tokens=4096','max_total_num_tokens=8192')))
        diff=identity.compare(a,b)
        self.assertEqual(diff['differing_fields'],[dict(field='scheduler_capacity.max_total_num_tokens',left=4096,right=8192)])
        self.assertIn('prefill_attention_backend',diff['null_or_auto_startup_args'])
        self.assertFalse(diff['production_authorized'])

    def test_missing_and_explicitly_unset_are_not_equivalent(self):
        args=dict(ARGS); del args['decode_attention_backend']
        result=identity.compare(identity.project(log()),identity.project(log(args,pool='')))
        self.assertIn('startup_args.decode_attention_backend',result['unobserved_fields'])
        self.assertIn('kv_pools.full',result['unobserved_fields'])
        self.assertEqual(result['differing_fields'],[])

    def test_no_false_integer_or_unbounded_opaque_value(self):
        r=identity.project(log(dict(ARGS,tp_size=True,reasoning_parser='https://internal.example/path')))
        self.assertEqual(r['unavailable_startup_args']['tp_size'],'unexpected_type_or_unset')
        self.assertEqual(r['unavailable_startup_args']['reasoning_parser'],'noncategorical_value_omitted')
        self.assertNotIn('internal.example',json.dumps(r))

    def test_refuses_ambiguous_instance_or_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            identity.project(log(ARGS,START+repr(ARGS)))
        with self.assertRaisesRegex(ValueError,'Conflicting runtime scheduler'):
            identity.project(log(ARGS,CAP.replace('context_len=1024','context_len=2048')))
        with self.assertRaisesRegex(ValueError,'Conflicting data'):
            identity.project(log(ARGS,KV.replace('bfloat16','float16')))
        with self.assertRaisesRegex(ValueError,'unique strings'):
            identity.project((START+"{'tp_size': 1, 'tp_size': 2}\n").encode())
