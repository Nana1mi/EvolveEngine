"""Quality-only inventory observer, in the candidate process (not a security boundary)."""
from pathlib import Path
from runtime_residency_probe import capture

_PHASE = Path('/ee-phase')
_seen = set()


def loaded(model):
    capture(model, 'after_load', strict=False)


def at_forward(model):
    for stage in ('ready', 'post_requests'):
        if stage not in _seen and (_PHASE / stage).is_file():
            _seen.add(stage)
            capture(model, stage, strict=False)


def instrument():
    runner = Path('/patched/python/sglang/srt/model_executor/model_runner.py')
    text = runner.read_text(encoding='utf-8')
    anchor = '        self.load_model()\n'
    if text.count(anchor) != 1:
        raise ValueError('No unique load checkpoint')
    text = text.replace(anchor, anchor + '        from residency_hook import loaded\n'
                                       '        loaded(self.model)\n')
    compile(text, str(runner), 'exec')
    (Path('/output') / 'model_runner.py').write_text(text, encoding='utf-8')

    model = Path('/patched/python/sglang/srt/models/spark2_5.py')
    text = model.read_text(encoding='utf-8')
    import ast
    cls, = (n for n in ast.parse(text).body if getattr(n, 'name', '') == 'Spark2_5ForCausalLM')
    fwd, = (n for n in cls.body if getattr(n, 'name', '') == 'forward')
    lines = text.splitlines(keepends=True)
    i = fwd.body[0].lineno - 1
    lines[i:i] = ['        from residency_hook import at_forward\n', '        at_forward(self)\n']
    text = ''.join(lines)
    compile(text, str(model), 'exec')
    (Path('/output') / 'spark2_5.py').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    instrument()
