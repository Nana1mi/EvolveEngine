"""G1-D3: split only the first named QKV layer; other layers use the exact QKV reference."""
from hashlib import sha256
from pathlib import Path
import sys

import reference_qkv as q

TARGET = 'model.layers.0.self_attn.q_k_v_proj'
REFERENCE = 'a5c3e5157bb8cfba8a5db700c21ac7ce226146648c676f3d30680e31d57c4b0b'
SELECT = '''                _name = layer._ee_g1d3_prefix
                if _name == "model.layers.0.self_attn.q_k_v_proj":
                    left = fused_mul_mat_gguf(x[:, :1024].contiguous(), qweight[:, :1088].contiguous(), WeightType.Q8_0)
                    right = fused_mul_mat_gguf(x[:, 1024:].contiguous(), qweight[:, 1088:].contiguous(), WeightType.Q8_0)
                    out = (left.float() + right.float()).to(x.dtype)
                    _path = "split"
                else:
                    out = fused_mul_mat_gguf(x, qweight, WeightType.Q8_0)
                    _path = "whole"
                from pathlib import Path as _Path
                _active = _Path('/tmp/ee-g1-numeric-active')
                if _active.is_file():
                    import hashlib as _hashlib
                    print('EE_G1D3_HIT', _active.read_text(), x.shape[0], _name, _path, __file__,
                          _hashlib.sha256(_Path(__file__).read_bytes()).hexdigest(), flush=True)
'''


def changed(original, role):
    if sha256(original).hexdigest() != q.BASE_SHA256:
        raise ValueError('base source mismatch')
    if role == 'anchor':
        return original
    if role not in ('local0a', 'local0b'):
        raise ValueError('unregistered role')
    ref = q.edited(original)
    factory = b'            return GGUFLinearMethod(self)\n'
    if sha256(ref).hexdigest() != REFERENCE or ref.count(factory) != 1 or ref.count(q.CALL.encode()) != 1:
        raise ValueError('invalid reference or edit anchors')
    tagged = ref.replace(factory, b'            layer._ee_g1d3_prefix = prefix\n' + factory)
    result = tagged.replace(q.CALL.encode(), SELECT.encode())
    compile(result, 'gguf.py', 'exec')
    return result


if __name__ == '__main__':
    path = Path('/patched') / q.RELATIVE
    data = changed(path.read_bytes(), sys.argv[1])
    if sys.argv[1] != 'anchor':
        path.write_bytes(data)
    print('EE_G1D3_SOURCE', sys.argv[1], sha256(path.read_bytes()).hexdigest(), flush=True)
