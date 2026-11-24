"""Construct a scoped diagnostic variant from the pinned production-source B reference."""
import sys
from hashlib import sha256
from pathlib import Path
import reference_qkv as q

HIT = '''                from pathlib import Path as _HitPath
                _tagfile = _HitPath('/tmp/ee-g1-numeric-active')
                if _tagfile.is_file():
                    _tag = _tagfile.read_text()
                    if _tag and getattr(self, '_ee_last_tag', None) != _tag:
                        import hashlib as _hh
                        print('EE_G1D_HIT', _tag, x.shape[0], __file__, _hh.sha256(_HitPath(__file__).read_bytes()).hexdigest(), flush=True)
                        self._ee_last_tag = _tag
'''
C='''                left = fused_mul_mat_gguf(x[:, :1024].contiguous(), qweight[:, :1088].contiguous(), WeightType.Q8_0)
                right = fused_mul_mat_gguf(x[:, 1024:].contiguous(), qweight[:, 1088:].contiguous(), WeightType.Q8_0)
                out = (left.float() + right.float()).to(x.dtype)
'''
D=q.CALL+'''                out[:, :2] = out[:, [1, 0]]
'''

def changed(original, role):
    assert sha256(original).hexdigest()==q.BASE_SHA256
    if role in ('e0a','e0b'): return original
    ref=q.edited(original); assert sha256(ref).hexdigest()=='a5c3e5157bb8cfba8a5db700c21ac7ce226146648c676f3d30680e31d57c4b0b' and ref.count(q.CALL.encode())==1
    data=ref.replace(q.CALL.encode(),({'b':q.CALL,'c':C,'d':D}[role]+HIT).encode()); compile(data,'gguf.py','exec'); return data

if __name__=='__main__':
    p=Path('/patched')/q.RELATIVE; original=p.read_bytes(); role=sys.argv[1]; data=changed(original,role)
    if role not in ('e0a','e0b'):p.write_bytes(data)
    print('EE_G1D_SOURCE',role,sha256(p.read_bytes()).hexdigest(),flush=True)
