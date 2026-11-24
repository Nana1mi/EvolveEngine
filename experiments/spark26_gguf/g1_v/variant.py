'''Install only the frozen predeclared branch into a fresh isolate.'''
from hashlib import sha256
from pathlib import Path
import json
import sys
import reference_qkv as q
import sampler_variant as sampler

INJECT=b'            return GGUFLinearMethod(self)\n'
HIT='''                _name = layer._ee_g1v_prefix
                _fault = _name == "model.layers.0.self_attn.q_k_v_proj" and FAULT
                if _fault:
                    out[:, 2560] = 0
                from pathlib import Path as _HitPath
                _tag = _HitPath('/tmp/ee-g1-numeric-active')
                if _tag.is_file():
                    from hashlib import sha256 as _srcsha
                    print('EE_G1V_HIT', _tag.read_text(), x.shape[0], _name, 'fault' if _fault else 'whole',
                          _srcsha(_HitPath(__file__).read_bytes()).hexdigest(), flush=True)
'''


def gguf_changed(raw, role):
    assert sha256(raw).hexdigest()==q.BASE_SHA256
    if role not in ('exact','faultv'): return raw
    ref=q.edited(raw); c=q.CALL.encode(); assert ref.count(c)==ref.count(INJECT)==1
    ref=ref.replace(INJECT,b'            layer._ee_g1v_prefix = prefix\n'+INJECT)
    edited=ref.replace(c,c+HIT.replace('FAULT',str(role=='faultv')).encode());compile(edited,'gguf.py','exec');return edited


if __name__=='__main__':
    role=sys.argv[1];plan=json.loads(Path('/diag/plan.json').read_bytes());assert role in plan['roles']
    gp=Path('/patched')/q.RELATIVE; sp=sampler.FILE; orig=sp.read_bytes(); assert sha256(orig).hexdigest()==plan['sampler_sha']
    gg=gguf_changed(gp.read_bytes(),role)
    if role in ('exact','faultv'):gp.write_bytes(gg)
    if role=='fp64greedy':
        out=sampler.changed(orig,plan['sampler_sha']);assert sha256(out).hexdigest()==plan['variant_sha'];sp.write_bytes(out)
    print('EE_G1V_SOURCE',role,sha256(gp.read_bytes()).hexdigest(),sha256(sp.read_bytes()).hexdigest(),flush=True)
