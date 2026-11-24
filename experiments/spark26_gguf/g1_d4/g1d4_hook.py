"""Shadow measurement; this module never supplies an alternate model output."""
from hashlib import sha256
from pathlib import Path
import inspect
import json
import math
import struct


def q8row(raw):
    if len(raw) != 2176:
        raise ValueError('unexpected packed length')
    for scale, *values in struct.iter_unpack('<e32b', raw):
        if not math.isfinite(scale):
            raise ValueError('nonfinite Q8_0 scale')
        yield from (scale * v for v in values)


def dot(x, w):
    lo = math.fsum(a*b for a,b in zip(x[:1024], w[:1024], strict=True))
    hi = math.fsum(a*b for a,b in zip(x[1024:], w[1024:], strict=True))
    return (lo, hi, math.fsum((lo, hi)))


def bf16(v):
    # Exact reference conversion from FP64, ties to even, for finite normal values in the tested range.
    if v == 0:
        return v
    _, exp = math.frexp(v)
    quantum = math.ldexp(1.0, max(-133, exp-8))
    return round(v / quantum) * quantum


def hash_tensor(t):
    return sha256(t.detach().contiguous().view(-1).cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def compare(x, weight, whole, fn, qtype, tag, source):
    import torch
    plan = json.loads(Path('/diag/plan.json').read_text())
    assert x.shape[1] == 2048 and tuple(weight.shape) == (3072,2176)
    assert x.dtype == whole.dtype == torch.bfloat16 and weight.dtype == torch.uint8
    assert list(whole.shape) == [x.shape[0], 3072]
    src = Path(source)
    left = fn(x[:, :1024].contiguous(), weight[:, :1088].contiguous(), qtype)
    right = fn(x[:, 1024:].contiguous(), weight[:, 1088:].contiguous(), qtype)
    assert left.dtype == right.dtype == whole.dtype and left.shape == right.shape == whole.shape
    summed = left.float() + right.float()
    split = summed.to(x.dtype)
    diff = (split.float() - whole.float()).abs()
    err = (summed - whole.float()).abs()
    ids = [0, x.shape[0]//2, x.shape[0]-1]
    cols = plan['channels']
    a = x[ids].float().cpu().tolist()
    ws = [list(q8row(bytes(w))) for w in weight[cols].cpu().tolist()]
    outputs = [t[ids][:,cols].float().cpu().tolist() for t in (left,right,whole,split)]
    rows = []
    for i, r in enumerate(ids):
        for j, c in enumerate(cols):
            rlo,rhi,rall = dot(a[i], ws[j])
            rows.append([r,c,rlo,rhi,rall,bf16(rlo),bf16(rhi),bf16(rall),bf16(bf16(rlo)+bf16(rhi)),*(t[i][j] for t in outputs)])
    j = dict(tag=tag, layer=plan['layer'], x=list(x.shape), w=list(weight.shape), dtype=str(x.dtype),
             x_sha=hash_tensor(x), weight_sha=hash_tensor(weight), source_sha=sha256(src.read_bytes()).hexdigest(),
             dispatcher=inspect.getsource(fn), pack_byte_boundary=1088, activation_boundary=1024,
             nonzero=int(torch.count_nonzero(diff).item()), count=diff.numel(), max_abs=float(diff.max().item()),
             mean_abs=float(diff.mean().item()), pre_round_abs_max=float(err.max().item()), reference=rows)
    print('EE_G1D4_LAYER '+json.dumps(j,separators=(',',':'),allow_nan=False),flush=True)

