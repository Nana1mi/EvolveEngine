import sys
from pathlib import Path
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import deterministic_probe
    import gguf_trace_audit
    import reference_coalesce
    import reference_profile_audit
    import runtime_residency_probe
    import runtime_residency_audit


class TestFollowupAudits(unittest.TestCase):
    def test_correlate_operator_gpu_launch_without_attributing_other_kernel(self):
        trace = {'traceEvents': [
            dict(ph='X', cat='cpu_op', name='sgl_kernel::ggml_mul_mat_vec_a8', args={'External id': 1, 'Input Dims': [[512, 2176], [2, 2048], [], []]}),
            dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', args={'External id': 1, 'correlation': 5}),
            dict(ph='X', cat='kernel', name='synthetic_mmvq', dur=80, args={'correlation': 5}),
            dict(ph='X', cat='kernel', name='unrelated', dur=20, args={'correlation': 42}),
        ]}
        audited = gguf_trace_audit.summarize_trace(trace)
        self.assertEqual(audited['linked_operator_kernels']['ggml_mul_mat_vec_a8'], {'synthetic_mmvq': 1})
        self.assertEqual(audited['kernels'][0]['partial_share_pct'], 80)
        self.assertEqual([(r['m'],r['n'],r['calls']) for r in audited['shape_linked_kernel_launches']], [(2,512,1)])
        self.assertEqual(gguf_trace_audit.FILENAME.fullmatch('short1-2.3-tp-0-extend.trace.json.gz').groups(),
                         ('short1', 'extend'))

    def test_residency_needs_all_checkpoints_and_expected_bytes(self):
        with self.assertRaisesRegex(ValueError, 'checkpoints'):
            runtime_residency_audit.audit('a bare log')
        packed, norms = runtime_residency_probe.expected_layout()
        self.assertEqual((len(packed), len(norms)), (141, 57))
        self.assertEqual(sum(n*(k//32)*34 for (n, k), _ in packed.values()), 1814261760)

    def test_deterministic_probe_requires_valid_output_ids(self):
        check = deterministic_probe.validate_finish
        self.assertEqual(check(dict(finish_reason={'type':'stop'}, completion_tokens=3), [1,2,3])[1], 3)
        with self.assertRaises(ValueError):
            check(dict(finish_reason={'type':'length'}, completion_tokens=3), [1,2])
        with self.assertRaises(ValueError):
            check(dict(finish_reason={'type':'stop'}, completion_tokens=1), [131072])

    def test_reference_patch_checks_raw_source(self):
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            reference_coalesce.edited(reference_coalesce.ANCHOR.encode())

    def test_profile_groups_normalize_on_unmodified_attention_gate(self):
        a = reference_profile_audit
        rows = [dict(m=1,n=n,k=k,calls=c,sum_us=dur,kernel='void mul_mat_vec_q<x>')
                for n,k,c,dur in [(8,2048,56,20),(3072,2048,56,56),(13312,2048,56,112)]]
        r = a.summarize_arm(rows, 1)
        self.assertEqual(r['forward_count'],2)
        self.assertEqual(r['groups']['ffn_gate_up']['matmul_us_per_forward'],56)
        self.assertEqual(r['groups']['qkv_plus_attention_out']['matmul_us_per_forward'],28)
        with self.assertRaisesRegex(ValueError,'anchor'):
            a.summarize_arm(rows[1:],1)
