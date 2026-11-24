"""Source-locked, development-only mathematically equivalent greedy sampler variant."""
import json
from hashlib import sha256
from pathlib import Path

FILE = Path('/patched/python/sglang/srt/layers/sampler.py')
A = "            _trace_e2e_sampler(\"greedy_enter\")\n            if _use_aiter and not _disable_aiter_greedy_sample:\n"
B = "                original_logprobs = logprobs = torch.nn.functional.log_softmax(\n                    logits, dim=-1\n                )\n"
C = "            _trace_e2e_sampler(\"greedy_enter\")\n"
H = """            from pathlib import Path as _EEPath
            _ee_tag = _EEPath('/tmp/ee-g1-numeric-active')
            _ee_high = None
            if (logprob_logits is None and logits.is_cuda and logits.dtype == torch.float32
                    and logits.shape == (1, 131072) and _ee_tag.is_file()):
                from hashlib import sha256 as _ee_sha256
                _ee_high = logits.double()
                batch_next_token_ids = torch.max(_ee_high, dim=-1).indices
                print('EE_G1D5_SAMPLER', _ee_tag.read_text(), str(logits.dtype),
                      tuple(logits.shape), int(batch_next_token_ids.item()),
                      _ee_sha256(_EEPath(__file__).read_bytes()).hexdigest(), flush=True)
"""
LO = """                original_logprobs = logprobs = (
                    torch.nn.functional.log_softmax(_ee_high, dim=-1).to(logits.dtype)
                    if _ee_high is not None else torch.nn.functional.log_softmax(logits, dim=-1)
                )
"""

def changed(data, expected):
    assert sha256(data).hexdigest() == expected
    s = data.decode('utf8'); assert s.count(A) == s.count(B) == 1
    out = s.replace(A,C+H+'            elif _use_aiter and not _disable_aiter_greedy_sample:\n').replace(B,LO)
    compile(out,'sampler.py','exec'); return out.encode('utf8')

if __name__=='__main__':
    FILE.write_bytes(changed(FILE.read_bytes(),json.loads(Path('/diag/plan.json').read_text())['sampler_sha']))
    print('EE_G1D5_SOURCE',sha256(FILE.read_bytes()).hexdigest(),flush=True)
