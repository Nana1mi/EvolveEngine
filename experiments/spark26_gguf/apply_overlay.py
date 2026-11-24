"""Apply the compatibility overlay to a fresh copy of the pinned SGLang tree."""
import argparse
from hashlib import sha256
from pathlib import Path

PIN = 'b2ce7f8811bc40fd0d67000030c22edbbe0f2a37'


def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f'Unexpected upstream source around {old.splitlines()[-1]}')
    return s.replace(old, new, 1)


def apply(root):
    srt = root / 'python/sglang/srt'
    loader = srt / 'model_loader/loader.py'
    model = srt / 'models/spark2_5.py'
    helper = srt / 'model_loader/spark2_5_gguf.py'
    if helper.exists():
        raise ValueError('Use a fresh checkout; overlay target already exists')
    originals = [p.read_bytes().replace(b'\r\n', b'\n') for p in (loader, model)]
    expected = (
        '6f096addfa32059c427286b5472db1d60ab50975a22e0945120b3e18ff944f80',
        '5146a992518a8abc63d0b7f9619f48189a4a374aa71ffa692df30c0a79b032b3',
    )
    if tuple(sha256(data).hexdigest() for data in originals) != expected:
        raise ValueError(f'Expected clean source files from {PIN}')
    l, m = (data.decode('utf-8') for data in originals)
    l = once(l, '''        local_model_path = self._prepare_weights(model_config.model_path)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
''', '''        local_model_path = self._prepare_weights(model_config.model_path)
        spark_gguf = None
        if model_config.hf_config.model_type == "spark2_5":
            from .spark2_5_gguf import Spark26GGUF

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
                raise ValueError("Spark Q8_0 compatibility currently requires a single device")
            spark_gguf = Spark26GGUF(local_model_path, model_config.hf_config)
            gguf_weights_map = {}
        else:
            gguf_weights_map = self._get_gguf_weights_map(model_config)
''')
    l = once(l, '''        if "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
''', '''        if spark_gguf is None and "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
''')
    l = once(l, '''            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map)
            )
''', '''            if spark_gguf is not None:
                model.load_weights(spark_gguf.weights(), strict_gguf=True)
            else:
                model.load_weights(self._get_weights_iterator(local_model_path, gguf_weights_map))
''')
    m = once(m, '''                quant_config=None,  # g_proj keeps bf16.
''', '''                # This particular GGUF stores the head-wise gate as Q8_0.
                quant_config=(quant_config if quant_config is not None
                              and quant_config.get_name() == "gguf" else None),
''')
    m = once(m, '''    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
''', '''    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], *, strict_gguf=False):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if strict_gguf:
            stacked_params_mapping += [(".q_k_v_proj", f".{s}_proj", s) for s in "qkv"]
''')
    m = once(m, '''            else:
                logger.warning(f"Parameter {name} not found in params_dict")
''', '''            elif strict_gguf:
                raise ValueError(f"Unrecognized Spark GGUF parameter {name}")
            else:
                logger.warning(f"Parameter {name} not found in params_dict")
''')
    replacement = Path(__file__).with_name(helper.name).read_text(encoding='utf-8')
    for p, code in [(loader, l), (model, m), (helper, replacement)]:
        compile(code, str(p), 'exec')
    for p, code in [(loader, l), (model, m), (helper, replacement)]:
        p.write_text(code, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fresh_copy', type=Path)
    a = p.parse_args()
    apply(a.fresh_copy)
