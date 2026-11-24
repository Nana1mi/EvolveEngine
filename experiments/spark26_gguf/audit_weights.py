"""Check tensor inventory and representative rows against the source checkpoint."""
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
from safetensors import safe_open

from spark2_5_gguf import Spark26GGUF


def audit(gguf_path, hf):
    config = SimpleNamespace(**json.loads((hf / 'config.json').read_text()))
    src = Spark26GGUF(gguf_path, config)
    weight_map = json.loads((hf / 'model.safetensors.index.json').read_text())['weight_map']
    if set(weight_map) != {v.reference + '.weight' for v in src.specs.values()}:
        raise ValueError('Reference checkpoint inventory differs')
    worst = 0.0
    rows = 0
    with ExitStack() as stack:
        files = {f: stack.enter_context(safe_open(str(hf / f), framework='pt', device='cpu'))
                 for f in set(weight_map.values())}
        for name, spec in src.specs.items():
            key = spec.reference + '.weight'
            ref = files[weight_map[key]].get_slice(key)
            if tuple(ref.get_shape()) != spec.shape:
                raise ValueError(f'Reference shape differs: {key}')
            tensor = src.tensors[name]
            if not spec.quantized:
                if not np.array_equal(tensor.data, ref[:].float().numpy()):
                    raise ValueError(f'F32 normalization tensor differs: {name}')
                continue
            n = spec.shape[0]
            labels = {0, n // 4, n // 2, 3 * n // 4, n - 1}
            if name.endswith('.attn_qkv.weight'):
                q, k = src.q_rows, src.kv_rows
                labels.update([q - 1, q, q + k - 1, q + k])
            for i in sorted(labels):
                expected = ref[i:i+1].float().numpy()
                actual = gguf.quants.dequantize(tensor.data[i:i+1], tensor.tensor_type)
                err = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
                if not np.isfinite(err) or err > 0.015:
                    raise ValueError(f'Source alignment failed: {name}, row {i}, rel_L2={err:.6g}')
                worst = max(worst, float(err))
                rows += 1
    return {'tensors_checked': len(src.specs), 'sampled_quantized_rows': rows, 'worst_relative_l2': worst}


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1], Path(sys.argv[2]))))
