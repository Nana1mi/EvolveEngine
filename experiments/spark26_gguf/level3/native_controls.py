"""Deterministic reference-only Native controls; never ship them to the Agent.

Generation does not execute the engine, approve numeric/residency thresholds,
or prove the changed kernel executed. Record controls before any measured run.
"""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from evolveengine.v3_level3_build import AOT_GGUF, ENGINE_GGUF, SGLANG_ARCHIVE_SHA256, SGLANG_REVISION
from evolveengine.v3_level3_source import SourceContract

MMVQ = AOT_GGUF.rsplit('/',1)[0]+'/mmvq.cuh'
KERNEL_WITNESS = 'ee_level3_positive_mmvq_control'
MMQ = AOT_GGUF.rsplit('/',1)[0]+'/mmq.cuh'
STORE = '      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];'
FAULT = '''      if constexpr (std::is_same<block_q_t, block_q8_0>::value) {
        dst[col_dst * nrows_dst + row_dst] = -sum[i / WARP_SIZE_GGUF][j / nwarps];
      } else {
        dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
      }'''
MIRROR_ANCHOR = '        shard_id = layer.qweight.shard_id\n\n        if shard_id:\n'
MIRROR = '''        shard_id = layer.qweight.shard_id

        if (type(layer).__name__ == "QKVParallelLinear"
                and not hasattr(layer, "_fixture_bf16_weight_copy")):
            layer._fixture_bf16_weight_copy = dequantize_gguf_weight(
                layer.qweight, WeightType.Q8_0, torch.bfloat16
            )

        if shard_id:
'''


def once(src, anchor, edit):
    if src.count(anchor)!=1:
        raise ValueError('expected one fixture anchor in pinned E0')
    return src.replace(anchor,edit)


def make(contract):
    """Return actual E0-relative candidates with three distinct causal controls."""
    e0 = contract.from_args({'files':{}})
    mode = contract.provenance
    if (mode.get('revision'),mode.get('archive_sha256')) != (SGLANG_REVISION,SGLANG_ARCHIVE_SHA256):
        raise ValueError('reference controls require an anchored source contract')
    files = contract.base
    header = files[MMVQ].decode('utf-8')
    if KERNEL_WITNESS in header or header.count('static __global__ void mul_mat_vec_q(')!=1:
        raise ValueError('unexpected pinned AOT kernel declaration')
    edited_header,n = re.subn(r'\bmul_mat_vec_q\b',KERNEL_WITNESS,header)
    if n < 2:
        raise ValueError('AOT fixture does not replace real CUDA callsites')
    prefill = files[MMQ].decode('utf-8')
    if ('typename block_q_t,' not in prefill or 'vec_dot_q8_0_q8_1_mul_mat>' not in prefill
            or 'static void ggml_mul_mat_q8_0_q8_1_cuda(' not in prefill):
        raise ValueError('missing pinned CUDA Q8_0 prefill path')
    # The earlier fixture added a separate CUDA launch and raised the ready
    # memory sample. Perturb the Q8_0 result in the existing kernel store.
    compute_negative = '#include <type_traits>\n'+once(prefill,STORE,FAULT)
    mirror = once(files[ENGINE_GGUF].decode('utf-8'),MIRROR_ANCHOR,MIRROR)
    compile(mirror,ENGINE_GGUF,'exec')
    variants = dict(e0=e0)
    for name,path,src in (('native_positive',MMVQ,edited_header),('native_compute_negative',MMQ,compute_negative),
                          ('residency_negative',ENGINE_GGUF,mirror)):
        variants[name] = contract.from_args({'files':{path:src}})
    if len({e.digest for e in variants.values()})!=len(variants):
        raise ValueError('controls do not identify distinct effective trees')
    return variants
