from hashlib import sha256
from pathlib import Path
import reference_qkv as q

TARGET = 'model.layers.0.self_attn.q_k_v_proj'
REFERENCE = 'a5c3e5157bb8cfba8a5db700c21ac7ce226146648c676f3d30680e31d57c4b0b'
SELECT = (
    q.CALL + '                if layer._ee_g1d4_prefix == '+repr(TARGET)+':\n'
    '                    from pathlib import Path as _Path\n'
    '                    _active = _Path("/tmp/ee-g1-numeric-active")\n'
    '                    if _active.is_file():\n'
    '                        from g1d4_hook import compare as _shadow\n'
    '                        _shadow(x, qweight, out, fused_mul_mat_gguf, WeightType.Q8_0,\n'
    '                                _active.read_text(), __file__)\n'
)


def changed(base):
    assert sha256(base).hexdigest() == q.BASE_SHA256
    ref = q.edited(base)
    factory = b'            return GGUFLinearMethod(self)\n'
    call = q.CALL.encode()
    assert sha256(ref).hexdigest() == REFERENCE and ref.count(factory) == ref.count(call) == 1
    tagged = ref.replace(factory, b'            layer._ee_g1d4_prefix = prefix\n' + factory)
    result = tagged.replace(call, SELECT.encode())
    compile(result, 'gguf.py', 'exec')
    return result


if __name__ == '__main__':
    path = Path('/patched') / q.RELATIVE
    path.write_bytes(changed(path.read_bytes()))
    print('EE_G1D4_SOURCE', sha256(path.read_bytes()).hexdigest(), flush=True)

