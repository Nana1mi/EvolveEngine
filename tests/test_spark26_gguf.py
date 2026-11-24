import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.spark26_gguf.spark2_5_gguf import Spark26GGUF, specs


def config(**overrides):
    props = dict(model_type='spark2_5', tie_word_embeddings=True,
                 num_hidden_layers=28, hidden_size=2048, intermediate_size=6656,
                 vocab_size=131072, num_attention_heads=8, num_key_value_heads=2,
                 head_dim=256)
    return SimpleNamespace(**(props | overrides))


class Spark26LayoutTests(unittest.TestCase):
    def test_exact_tied_inventory_and_quantized_gate(self):
        m = specs(config())
        self.assertEqual(len(m), 226)
        self.assertEqual(sum(x.quantized for x in m.values()), 169)
        self.assertNotIn('output.weight', m)
        self.assertEqual(m['token_embd.weight'].reference, 'model.embedding')
        self.assertEqual(m['blk.0.attn_qkv.weight'].shape, (3072, 2048))
        self.assertTrue(m['blk.27.attn_gate.weight'].quantized)

    def test_unknown_model_size_or_binding_is_rejected(self):
        for k, v in [('model_type', 'qwen3'), ('num_hidden_layers', 24), ('tie_word_embeddings', False)]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                specs(config(**{k:v}))

    def test_fused_qkv_submits_explicit_packed_shards(self):
        adapter = Spark26GGUF.__new__(Spark26GGUF)
        name = 'blk.0.attn_qkv.weight'
        adapter.specs = {name: specs(config())[name]}
        adapter.q_rows, adapter.kv_rows = 2048, 512
        adapter.tensors = {name: SimpleNamespace(tensor_type=8, data=range(3072))}
        torch = SimpleNamespace(tensor=lambda x, **_: x, uint8=object())
        with patch.dict(sys.modules, {'torch':torch}):
            rows = list(adapter.weights())
        head = 'model.layers.0.self_attn.'
        for i, tag, bounds in [(0,'q',(0,2048)), (1,'k',(2048,2560)), (2,'v',(2560,3072))]:
            self.assertEqual(rows[2*i], (head + tag + '_proj.qweight_type', 8))
            self.assertEqual(rows[2*i+1], (head + tag + '_proj.qweight', range(*bounds)))


if __name__ == '__main__':
    unittest.main()
