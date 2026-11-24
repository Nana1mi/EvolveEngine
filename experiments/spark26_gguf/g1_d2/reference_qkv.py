"""Restore E0 FFN, and join only the exact Spark TP1 QKV layout and tested M values."""
import argparse
from hashlib import sha256
from pathlib import Path

BASE_SHA256 = '4d3f9f42cd531e4e3c0542106a91081f6fddd7396971013a413722970cf851ec'
RELATIVE = Path('python/sglang/srt/layers/quantization/gguf.py')
ANCHOR = '        shard_id = layer.qweight.shard_id\n\n        if shard_id:\n'
PROFILED_ROWS = (1, 2, 130, 260, 768, 1536)
CALL = '                out = fused_mul_mat_gguf(x, qweight, WeightType.Q8_0)\n'
GUARD = '''        shard_id = layer.qweight.shard_id

        if (_is_cuda and type(layer).__name__ == "QKVParallelLinear"
            and shard_id and tuple(shard_id) == ("q", "k", "v") and x.ndim == 2
            and x.shape[0] in ROWS and x.shape[1] == 2048
            and x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
        ):
            qweight = layer.qweight
            layout = {"q": (0, 2048, 2176), "k": (2048, 2560, 2176),
                      "v": (2560, 3072, 2176)}
            if (qweight.shape == (3072, 2176) and qweight.dtype == torch.uint8
                and qweight.device == x.device and qweight.is_contiguous()
                and qweight.shard_offset_map == layout
                and layer.qweight_type.shard_weight_type == dict.fromkeys(layout, WeightType.Q8_0)
            ):
                out = fused_mul_mat_gguf(x, qweight, WeightType.Q8_0)
                if bias is not None:
                    out.add_(bias)
                return out

        if shard_id:
'''
ORACLE = '''                m = x.shape[0]
                if m not in _ee_qkv_checked:
                    old = torch.cat([fused_mul_mat_gguf(
                        x, qweight[start:end, :stride].contiguous(), WeightType.Q8_0
                    ) for start, end, stride in layout.values()], dim=1)
                    if not torch.equal(old.view(torch.int16), out.view(torch.int16)):
                        raise RuntimeError(f"QKV_SHADOW_MISMATCH M={m}")
                    _ee_qkv_checked.add(m)
                    print(f"EE_COALESCE_EXACT QKVParallelLinear M={m}", flush=True)
'''


def edited(source, verify_exact=False):
    normalized = source.replace(b'\r\n', b'\n')
    if sha256(normalized).hexdigest() != BASE_SHA256:
        raise ValueError('Expected unchanged fixed-revision GGUF quantization source')
    text = normalized.decode('utf-8')
    if text.count(ANCHOR) != 1:
        raise ValueError('Expected unique upstream GGUF apply anchor')
    guard = GUARD.replace('in ROWS', f'in {PROFILED_ROWS!r}')
    if verify_exact:
        if guard.count(CALL) != 1:
            raise ValueError('Cannot insert shadow check')
        guard = guard.replace(CALL,CALL+ORACLE)
        cls = 'class GGUFLinearMethod(LinearMethodBase):'
        if text.count(cls) != 1:
            raise ValueError('Expected a unique target GGUF class')
        text = text.replace(cls,'_ee_qkv_checked = set()\n\n\n'+cls)
    out = text.replace(ANCHOR,guard)
    compile(out,str(RELATIVE),'exec')
    return out.encode('utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('disposable_root',type=Path)
    p.add_argument('--verify-exact',action='store_true')
    args = p.parse_args()
    root = args.disposable_root.resolve(strict=True)
    target = (root/RELATIVE).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError('Resolved file leaves the disposable root')
    data = edited(target.read_bytes(),args.verify_exact)
    target.write_bytes(data)
    print('ee_spark26_reference_qkv_sha256='+sha256(data).hexdigest(),flush=True)


if __name__=='__main__':
    main()
