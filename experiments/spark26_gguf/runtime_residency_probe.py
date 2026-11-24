"""Diagnostic only: inventory packed CUDA weights and live model-owned storages."""
import json
import os
from pathlib import Path
import types


Q8 = 8
STAMP = Path('/tmp/ee-spark26-residency-trigger')


def expected_layout():
    packed = {'model.embed_tokens': ((131072, 2048), None)}
    norms = {'model.norm'}
    for i in range(28):
        prefix = f'model.layers.{i}.'
        packed.update({prefix+n: ((rows, cols), shards) for n, rows, cols, shards in (
            ('self_attn.q_k_v_proj', 3072, 2048, {'q': 2048, 'k': 512, 'v': 512}),
            ('self_attn.g_proj', 8, 2048, None),
            ('self_attn.out_proj', 2048, 2048, None),
            ('mlp.gate_up_proj', 13312, 2048, {0: 6656, 1: 6656}),
            ('mlp.down_proj', 2048, 6656, None),
        )})
        norms.update(prefix+n for n in ('input_layernorm', 'post_attention_layernorm'))
    return packed, norms


def capture(model, label, *, strict=True):
    import torch

    torch.cuda.synchronize()
    modules = dict(model.named_modules(remove_duplicate=False))
    packed, norms = expected_layout()
    issues, inventory = [], {}
    declared = {}

    def describe(tensor):
        st = tensor.untyped_storage()
        return dict(device=str(tensor.device), dtype=str(tensor.dtype), shape=list(tensor.shape),
                    bytes=st.nbytes(), address=st.data_ptr(), offset=tensor.storage_offset(),
                    contiguous=tensor.is_contiguous())

    def admit(name, t, row, role):
        if not t.is_cuda or row['offset'] != 0 or not row['contiguous'] or row['address'] == 0:
            issues.append(f'{name}: expected contiguous nonempty CUDA storage without offset')
        if row['address'] in declared:
            issues.append(f'{name}: undesignated shared storage with {declared[row["address"]]}')
        declared[row['address']] = name
        inventory[name] = dict(role=role, **row)

    if modules.get('lm_head') is not modules.get('model.embed_tokens'):
        issues.append('lm_head is not tied to the same embedding module')
    actual = {name for name, m in modules.items() if getattr(m, 'qweight', None) is not None}
    if actual != {*packed, 'lm_head'}:
        issues.append(f'packed module names differ: missing {sorted(set(packed)-actual)}, extra {sorted(actual-set(packed)-{"lm_head"})}')
    for name, (logical, shards) in packed.items():
        m = modules[name]
        p = m.qweight
        kind = m.qweight_type
        row = describe(p)
        n, k = logical
        stride = k//32*34
        if p.dtype != torch.uint8 or tuple(p.shape) != (n, stride) or row['bytes'] != n*stride:
            issues.append(f'{name}: Q8_0 packed shape/dtype/storage mismatch')
        if shards is None:
            if kind.weight_type != Q8 or p.shard_id:
                issues.append(f'{name}: Q8_0 type/shard mismatch')
        else:
            start = 0
            for shard, count in shards.items():
                stop = start+count
                if (tuple(p.shard_offset_map.get(shard, ())) != (start, stop, stride)
                        or kind.shard_weight_type.get(shard) != Q8):
                    issues.append(f'{name}: wrong type/shape/order for shard {shard}')
                start = stop
            if set(p.shard_id) != set(shards) or set(p.shard_offset_map) != set(shards):
                issues.append(f'{name}: extra or missing shards')
        admit(name, p, row, 'Q8_0')
        if kind.is_cuda and kind.numel():
            admit(name+'.qweight_type', kind, describe(kind), 'gguf_type_metadata')
    for name in sorted(norms):
        w = modules[name].weight
        row = describe(w)
        if not w.is_floating_point() or tuple(w.shape) != (2048,) or row['bytes'] != w.numel()*w.element_size():
            issues.append(f'{name}: unexpected native norm layout')
        admit(name, w, row, 'native_float_norm')

    # The two registered RoPE tables (one per attention layout) are generated
    # data, not weights. The larger table has the embedding's element count.
    rope_groups = {}
    for i in range(28):
        cols = 64 if i % 4 == 3 else 256
        rope = modules[f'model.layers.{i}.self_attn.rotary_emb']
        table = rope._buffers['cos_sin_cache']
        if (table.dtype != torch.float32 or tuple(table.shape) != (1048576, cols)
                or table.device.type != 'cuda'):
            issues.append(f'rotary table at layer {i}: unexpected registered layout')
        prior = rope_groups.setdefault(cols, table)
        if table is not prior:
            issues.append(f'rotary table at layer {i}: unregistered duplicate storage')
    for cols, table in rope_groups.items():
        admit(f'shared_rotary_cos_sin_cache_{cols}', table, describe(table), 'derived_rope_cache')
    unclassified, seen = {}, set()
    floats = {n*k for (n, k), _ in packed.values()} | {2048*2048, 512*2048, 6656*2048}
    # Walk model-owned attributes as well as named parameters/buffers. No descent
    # into functions, their closures/globals, or C++ objects without Python attrs.
    def visit(obj, name):
        if id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, type,
                types.FunctionType, types.MethodType, types.ModuleType, types.BuiltinFunctionType)):
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            row = describe(obj)
            if row['address'] not in declared and row['bytes']:
                unclassified.setdefault(row['address'], []).append(dict(path=name, **row))
                if obj.is_floating_point() and obj.numel() in floats:
                    issues.append(f'{name}: possible resident floating-point weight mirror')
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f'{name}[{k!r}]' if isinstance(k, (str,int)) else name+'[object-key]')
        elif isinstance(obj, (tuple, list, set)):
            for i,v in enumerate(obj):
                visit(v, f'{name}[{i}]')
        elif hasattr(obj, '__dict__'):
            visit(vars(obj), name)

    visit(model, 'root')
    packed_bytes = sum(r['bytes'] for r in inventory.values() if r['role'] == 'Q8_0')
    data = dict(label=label, issues=issues, inventory=inventory, packed_bytes=packed_bytes,
                model_owned_unclassified=list(unclassified.values()),
                allocator=dict(live_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()),
                note='Model ownership and PyTorch allocator scope; cannot attribute opaque native/CUDA allocations or prove absence of every external cache.')
    print('EE_RESIDENCY '+json.dumps(data, sort_keys=True), flush=True)
    if os.environ.get('EE_SPARK26_ALLOCATOR_TRACE') == '1':
        from allocator_residency_trace import capture as capture_allocator
        capture_allocator(data)
    if strict and issues:
        raise RuntimeError(f'Residency checks failed at {label}: {issues}')


def if_invoked(model):
    if STAMP.is_file():
        stage = STAMP.read_text(encoding='utf-8').strip()
        STAMP.unlink()
        if stage not in ('ready', 'post_requests'):
            raise ValueError('Unexpected residency checkpoint')
        capture(model, stage)


def instrument():
    runner = Path('/patched/python/sglang/srt/model_executor/model_runner.py')
    text = runner.read_text(encoding='utf-8')
    if os.environ.get('EE_SPARK26_ALLOCATOR_TRACE') == '1':
        earlier = '        self.init_torch_distributed()\n'
        if text.count(earlier) != 1:
            raise ValueError('Unable to identify early allocation observation point')
        text = text.replace(earlier, '        from allocator_residency_trace import begin\n'
                                     '        begin()\n'+earlier)
    hook = '        self.load_model()\n'
    if text.count(hook) != 1:
        raise ValueError('Unable to identify sole model-load checkpoint')
    text = text.replace(hook, hook+'        from runtime_residency_probe import capture\n'
                                    '        capture(self.model, "after_load")\n')
    compile(text, str(runner), 'exec')
    runner.write_text(text, encoding='utf-8')
    path = Path('/patched/python/sglang/srt/models/spark2_5.py')
    text = path.read_text(encoding='utf-8')
    # Insert right before executing top-level model forward (not model.layers forward).
    import ast
    tree = ast.parse(text)
    cls, = (node for node in tree.body if getattr(node, 'name', '') == 'Spark2_5ForCausalLM')
    fwd, = (node for node in cls.body if getattr(node, 'name', '') == 'forward')
    lines = text.splitlines(keepends=True)
    i = fwd.body[0].lineno-1
    text = ''.join(lines[:i]+['        from runtime_residency_probe import if_invoked\n',
                             '        if_invoked(self)\n']+lines[i:])
    compile(text, str(path), 'exec')
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    instrument()
