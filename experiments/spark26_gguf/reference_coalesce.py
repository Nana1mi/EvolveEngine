"""Prepared, not accepted: coalesce exact Spark TP1 Q8_0 shard layouts for diagnosis."""
import argparse
from hashlib import sha256
from pathlib import Path

BASE_SHA256 = '4d3f9f42cd531e4e3c0542106a91081f6fddd7396971013a413722970cf851ec'
RELATIVE = Path('python/sglang/srt/layers/quantization/gguf.py')
ANCHOR = '        shard_id = layer.qweight.shard_id\n\n        if shard_id:\n'

# All shard boundaries are multiples of 128 output rows. Both each shard and the
# joined weight have the same side of the 5120-row MMVQ dispatch cutoff.
GUARD = '''        shard_id = layer.qweight.shard_id

        if shard_id and _is_cuda and isinstance(self, GGUFLinearMethod):
            qweight, qtypes = layer.qweight, layer.qweight_type.shard_weight_type
            layout = None
            if (type(layer).__name__, tuple(shard_id), tuple(qweight.shape)) == (
                "QKVParallelLinear", ("q", "k", "v"), (3072, 2176)
            ):
                layout = {"q": (0, 2048, 2176), "k": (2048, 2560, 2176),
                          "v": (2560, 3072, 2176)}
            elif (type(layer).__name__, tuple(shard_id), tuple(qweight.shape)) == (
                "MergedColumnParallelLinear", (0, 1), (13312, 2176)
            ):
                layout = {0: (0, 6656, 2176), 1: (6656, 13312, 2176)}
            if (layout is not None and qweight.dtype == torch.uint8
                and qweight.is_cuda and qweight.is_contiguous()
                and qweight.shard_offset_map == layout
                and qtypes == dict.fromkeys(layout, WeightType.Q8_0)
            ):
                out = fused_mul_mat_gguf(x, qweight, WeightType.Q8_0)
                if bias is not None:
                    out.add_(bias)
                return out

        if shard_id:
'''


def edited(source, verify_exact=False):
    normalized = source.replace(b'\r\n', b'\n')
    if sha256(normalized).hexdigest() != BASE_SHA256:
        raise ValueError('Expected unmodified fixed-revision GGUF quantization source')
    text = normalized.decode('utf-8')
    if text.count(ANCHOR) != 1:
        raise ValueError('Cannot identify exactly one GGUFLinearMethod.apply anchor')
    guard = GUARD
    if verify_exact:
        # Diagnostic path only. Never include shadow calls or their evidence in timing.
        line = '                out = fused_mul_mat_gguf(x, qweight, WeightType.Q8_0)\n'
        verify = '''                key = (type(layer).__name__, x.shape[0])
                if x.shape[0] in (1, 2, 130, 260, 768, 1536) and key not in _ee_coalesce_checked:
                    old = torch.cat([fused_mul_mat_gguf(
                        x, qweight[start:end, :stride].contiguous(), WeightType.Q8_0
                    ) for start, end, stride in layout.values()], dim=1)
                    if not torch.equal(old, out):
                        raise RuntimeError(f"COALESCE_SHADOW_MISMATCH {key}")
                    _ee_coalesce_checked.add(key)
                    print(f"EE_COALESCE_EXACT {key[0]} M={key[1]}", flush=True)
'''
        if guard.count(line) != 1:
            raise ValueError('Missing canonical candidate call')
        guard = guard.replace(line, line+verify)
        cls = 'class GGUFLinearMethod(LinearMethodBase):'
        if text.count(cls) != 1:
            raise ValueError('Missing canonical GGUF class definition')
        text = text.replace(cls, '_ee_coalesce_checked = set()\n\n\n'+cls)
    result = text.replace(ANCHOR, guard)
    compile(result, str(RELATIVE), 'exec')
    return result.encode('utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('disposable_root', type=Path)
    p.add_argument('--verify-exact', action='store_true', help='diagnostic-only shadow equality')
    args = p.parse_args()
    root = args.disposable_root.resolve(strict=True)
    target = (root/RELATIVE).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError('Resolved source must be in the disposable root')
    new = edited(target.read_bytes(), verify_exact=args.verify_exact)
    target.write_bytes(new)
    print('ee_spark26_reference_coalesce_sha256='+sha256(new).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
