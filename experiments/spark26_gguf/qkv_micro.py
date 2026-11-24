"""Offline native Q8_0 projection timing; same CUDA operator/weights, split vs joined."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

LOCK = Path(__file__).with_name('qkv_micro_lock.json')
MODEL = Path('/model')


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def packed_weights(checkpoint, kind, layer):
    import torch

    prefix = f'blk.{layer}.'
    if kind == 'qkv':
        packed = torch.tensor(checkpoint.tensors[prefix+'attn_qkv.weight'].data, device='cuda')
        rows = (2048,512,512)
    elif kind == 'gate_up_control':
        packed = torch.cat([torch.tensor(checkpoint.tensors[prefix+f'ffn_{s}.weight'].data)
                            for s in ('gate','up')]).to('cuda')
        rows = (6656,6656)
    else:
        raise ValueError(f'Unexpected projection {kind}')
    if tuple(packed.shape) != (sum(rows),2176) or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError('Unexpected packed allocation')
    splits = packed.split(rows, dim=0)
    if any(not s.is_contiguous() or s.untyped_storage().data_ptr()!=packed.untyped_storage().data_ptr() for s in splits):
        raise ValueError('Split views cannot add another weight allocation')
    return packed,splits


def run(lock, lock_digest):
    import torch
    from spark2_5_gguf import Spark26GGUF
    from sglang.srt.layers.quantization import gguf as quant

    code = Path(quant.__file__).resolve()
    if not code.is_relative_to('/patched/python') or sha256(code.read_bytes()).hexdigest() != lock['source_sha256']:
        raise ValueError('Wrong serving GGUF module')
    if torch.cuda.get_device_capability() != (12,1) or torch.cuda.device_count() != 1:
        raise ValueError('Expected one GB10 GPU')
    cfg = SimpleNamespace(**json.loads((MODEL/'config.json').read_bytes()))
    ckpt = Spark26GGUF(MODEL/'Spark-X2.6-1.7B-Q8_0.gguf', cfg)
    cuda_op = quant.fused_mul_mat_gguf
    weight_type = quant.WeightType.Q8_0
    emit(dict(kind='identity', lock_sha256=lock_digest, source=str(code), source_sha256=lock['source_sha256'],
              gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=str(torch.__version__)))

    def go(arm, x, item):
        full,slices = item
        if arm == 'a':
            return torch.cat([cuda_op(x, s.contiguous(), weight_type) for s in slices], dim=1)
        return cuda_op(x, full, weight_type)

    for kind, shapes in (('qkv',lock['shape_order']),('gate_up_control',lock['negative_control_shape_order'])):
        weights = [packed_weights(ckpt, kind, layer) for layer in lock['layers']]
        for m in shapes:
            rng = torch.Generator(device='cpu').manual_seed(lock['activation_seed']+m)
            inputs = [torch.randn((m,2048), dtype=torch.float32, generator=rng).to('cuda', dtype=torch.bfloat16)
                      for _ in weights]
            for lvl,x,w in zip(lock['layers'],inputs,weights):
                a,b = go('a',x,w),go('b',x,w)
                if a.dtype != torch.bfloat16 or b.shape != a.shape or not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    raise ValueError(f'BF16 bit pattern changed for {kind} M={m} layer={lvl}')
            del a,b
            emit(dict(kind='exact', projection=kind, m=m, layers=lock['layers']))

            def sweep(arm, count):
                for _ in range(count):
                    for x,w in zip(inputs, weights):
                        go(arm,x,w)

            for arm in 'ab':
                sweep(arm,lock['warm_sweeps_per_arm'])
            torch.cuda.synchronize()
            count = lock['sweeps_per_sample']
            for idx,order in enumerate(lock['sample_orders']):
                row = {}
                for arm in order:
                    # One stream, no per-call sync, one event pair per rotating sweep.
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    sweep(arm,count)
                    end.record()
                    end.synchronize()
                    row[arm] = start.elapsed_time(end)/(count*len(weights))
                emit(dict(kind='sample', projection=kind, m=m, index=idx, order=order, ms_per_call=row))
            del inputs
        del weights
    emit(dict(kind='complete'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    raw = LOCK.read_bytes()
    lock = json.loads(raw)
    import torch
    with torch.inference_mode():
        run(lock,sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
