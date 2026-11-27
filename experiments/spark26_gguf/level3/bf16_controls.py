"""Deterministic BF16 Native/JIT & weight-audit controls; never ship them to the Agent.

Provides three causal BF16 controls against the unmodified pinned SGLang tree:
- positive: legal FP32 identity in live JIT CUDA gelu_tanh + typed ee_runtime.json
- fault: live JIT CUDA numerical corruption in gelu_tanh (negated activation)
- lowprec: persistent FP8 target-weight copy in Spark2_5MLP (flagged by weight audit)
"""
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_build import SGLANG_ARCHIVE_SHA256, SGLANG_REVISION
from evolveengine.v3_level3_startup import PATH as RUNTIME_PATH

ACTIVATION_CU = 'python/sglang/kernels/jit/csrc/elementwise/activation.cuh'
SPARK_MODEL = 'python/sglang/srt/models/spark2_5.py'
GELU_ANCHOR = '    return device::activation::gelu_tanh<kFastMath>(x_f32);'
GELU_POSITIVE = '    return 1.0f * device::activation::gelu_tanh<kFastMath>(x_f32);'
GELU_FAULT = '    return -device::activation::gelu_tanh<kFastMath>(x_f32);'
MLP_ANCHOR = '        gate_up, _ = self.gate_up_proj(x)\n'
MLP_LOWPREC = (
    '        if not hasattr(self.gate_up_proj, "_fixture_fp8_weight_copy"):\n'
    '            self.gate_up_proj._fixture_fp8_weight_copy = self.gate_up_proj.weight.to(torch.float8_e4m3fn)\n'
    '        gate_up, _ = self.gate_up_proj(x)\n'
)
MLP_LONGCTX_FAULT = (
    '        gate_up, _ = self.gate_up_proj(x)\n'
    '        if x.shape[0] >= 4096:\n'
    '            gate_up = -gate_up\n'
)
POSITIVE_KNOBS = {'cuda-graph-bs-decode': [1, 2, 3, 4, 5, 6], 'max-running-requests': 6, 'radix-cache': True}
LONGCTX_POSITIVE_KNOBS = {'chunked-prefill-size': 4096, 'swa-full-tokens-ratio': 0.75, 'radix-cache': True}


def _once(src, anchor, replacement):
    if src.count(anchor) != 1:
        raise ValueError(f'expected exactly one BF16 control anchor: {anchor[:60]!r}')
    return src.replace(anchor, replacement, 1)


def make(contract, *, include_long_ctx=False):
    """Return E0-relative BF16 control candidates."""
    prov = contract.provenance
    if (prov.get('revision'), prov.get('archive_sha256'), prov.get('target_precision')) != (
        SGLANG_REVISION, SGLANG_ARCHIVE_SHA256, 'bf16'
    ):
        raise ValueError('BF16 controls require a pinned BF16 source contract')
    e0 = contract.from_args({'files': {}})
    act = contract.base[ACTIVATION_CU].decode('utf-8')
    model_py = contract.base[SPARK_MODEL].decode('utf-8')
    pos_act = _once(act, GELU_ANCHOR, GELU_POSITIVE)
    fault_act = _once(act, GELU_ANCHOR, GELU_FAULT)
    lowprec_py = _once(model_py, MLP_ANCHOR, MLP_LOWPREC)
    compile(lowprec_py, SPARK_MODEL, 'exec')
    variants = {
        'base': e0,
        'positive': contract.from_args({'files': {ACTIVATION_CU: pos_act, RUNTIME_PATH: canonical(POSITIVE_KNOBS).decode()}}),
        'fault': contract.from_args({'files': {ACTIVATION_CU: fault_act}}),
        'lowprec': contract.from_args({'files': {SPARK_MODEL: lowprec_py}}),
    }
    if include_long_ctx:
        longctx_fault_py = _once(model_py, MLP_ANCHOR, MLP_LONGCTX_FAULT)
        compile(longctx_fault_py, SPARK_MODEL, 'exec')
        variants['longctx_pos'] = contract.from_args({'files': {RUNTIME_PATH: canonical(LONGCTX_POSITIVE_KNOBS).decode()}})
        variants['longctx_fault'] = contract.from_args({'files': {SPARK_MODEL: longctx_fault_py}})
    if len({v.digest for v in variants.values()}) != len(variants):
        raise ValueError('BF16 controls do not identify distinct effective trees')
    return variants
