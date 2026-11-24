"""Native Q8_0 iterator for the tied-embedding, fused-QKV Spark-X2.6-1.7B layout."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    target: str
    reference: str
    shape: tuple
    quantized: bool = True


def specs(config):
    if config.model_type != "spark2_5" or not config.tie_word_embeddings:
        raise ValueError("Only the tied Spark GGUF layout is supported")
    dims = (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
            config.vocab_size, config.num_attention_heads, config.num_key_value_heads,
            config.head_dim)
    if dims != (28, 2048, 6656, 131072, 8, 2, 256):
        raise ValueError(f"Unsupported Spark GGUF dimensions: {dims}")
    layers, hidden, intermediate, vocab, heads, kv_heads, dim = dims
    result = {
        "token_embd.weight": TensorSpec("model.embed_tokens", "model.embedding", (vocab, hidden)),
        "output_norm.weight": TensorSpec("model.norm", "model.norm", (hidden,), False),
    }
    for i in range(layers):
        prefix = f"model.layers.{i}."
        for gguf_name, name, shape, quant in (
            ("attn_qkv", "self_attn.q_k_v_proj", ((heads + 2 * kv_heads) * dim, hidden), True),
            ("attn_gate", "self_attn.g_proj", (heads, hidden), True),
            ("attn_output", "self_attn.out_proj", (hidden, heads * dim), True),
            ("ffn_gate", "mlp.gate_proj", (intermediate, hidden), True),
            ("ffn_up", "mlp.up_proj", (intermediate, hidden), True),
            ("ffn_down", "mlp.down_proj", (hidden, intermediate), True),
            ("attn_norm", "input_layernorm", (hidden,), False),
            ("ffn_norm", "post_attention_layernorm", (hidden,), False),
        ):
            result[f"blk.{i}.{gguf_name}.weight"] = TensorSpec(prefix + name, prefix + name, shape, quant)
    return result


class Spark26GGUF:
    """Reject extras, missing weights and unsupported layouts before writing any parameter."""

    def __init__(self, path, config):
        import gguf

        self.specs = specs(config)
        reader = gguf.GGUFReader(path)
        self.reader = reader
        if (reader.get_field("general.architecture").contents() != "spark2_5"
                or reader.get_field("general.file_type").contents() != 7):
            raise ValueError("Expected Spark2.5 architecture and Q8_0 GGUF")
        self.tensors = {tensor.name: tensor for tensor in reader.tensors}
        if len(self.tensors) != len(reader.tensors) or self.tensors.keys() != self.specs.keys():
            raise ValueError(f"Unexpected GGUF inventory: missing={sorted(self.specs.keys()-self.tensors.keys())}, "
                             f"extra={sorted(self.tensors.keys()-self.specs.keys())}")
        for name, spec in self.specs.items():
            t = self.tensors[name]
            expected_type = "Q8_0" if spec.quantized else "F32"
            if tuple(reversed(t.shape.tolist())) != spec.shape or t.tensor_type.name != expected_type:
                raise ValueError(f"Unexpected shape/type for {name}: {t.shape}/{t.tensor_type.name}")
        self.q_rows = config.num_attention_heads * config.head_dim
        self.kv_rows = config.num_key_value_heads * config.head_dim

    def weights(self):
        import torch

        def packed(prefix, t, data):
            yield prefix + ".qweight_type", torch.tensor(int(t.tensor_type), dtype=torch.uint8)
            yield prefix + ".qweight", torch.tensor(data)

        for name, spec in self.specs.items():
            t = self.tensors[name]
            if name.endswith(".attn_qkv.weight"):
                prefix = spec.target.removesuffix("q_k_v_proj")
                q, k = self.q_rows, self.kv_rows
                for shard, rows in zip("qkv", (slice(0, q), slice(q, q + k), slice(q + k, q + 2 * k))):
                    # SGLang's native GGUF QKV loader requires explicit shard types/IDs.
                    yield from packed(prefix + shard + "_proj", t, t.data[rows])
            elif spec.quantized:
                yield from packed(spec.target, t, t.data)
            else:
                yield spec.target + ".weight", torch.tensor(t.data)
