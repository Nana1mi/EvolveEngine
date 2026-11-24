#!/usr/bin/env bash
set -euo pipefail
export PYTHONPYCACHEPREFIX=/tmp/ee-v3-pycache PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=/tmp/ee-v3-triton CUDA_CACHE_PATH=/tmp/ee-v3-cuda
python - <<'PY'
import hashlib,json
from pathlib import Path
s=json.loads(Path('/ee-input/patch.json').read_bytes());plan=json.loads(Path('/ee-input/plan.json').read_bytes())
assert set(s)=={'operation','engine','files','base','node_limit_ms'} and s['operation'] in ('develop_quality','develop_measure')
for path,original in s['base'].items():
    assert path in plan['allowed']; p=Path('/patched')/path
    expected=hashlib.sha256(s['files'][path].encode('utf8')).hexdigest() if path in s['files'] else original
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    compile(p.read_bytes(),path,'exec')
    print('EE_V3_ENGINE_SOURCE',path,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
assert set(s['files'])<=set(s['base'])
(Path('/tmp/ee-v3-service.sh')).write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+f"""exec python -m sglang.launch_server --model-path /model/Spark-X2.6-1.7B-Q8_0.gguf --load-format gguf --tokenizer-path /model --chat-template /model/chat_template.jinja --trust-remote-code --host 127.0.0.1 --port 30091 --served-model-name spark26-q8 --tp-size 1 --context-length {plan['context_length']} --max-total-tokens {plan['max_total_tokens']} --mem-fraction-static 0.4 --max-running-requests {plan['max_running_requests']} --attention-backend triton --disable-radix-cache --stream-interval 1 --disable-decode-cuda-graph --disable-prefill-cuda-graph --tool-call-parser spark25 --reasoning-parser qwen3
""")
PY
python - <<'PY'
import importlib.metadata as m
import sglang,torch
assert sglang.__file__.startswith('/patched/python/') and torch.cuda.is_available()
print('runtime',sglang.__file__,torch.__version__,torch.cuda.get_device_name(),*(f'{p}={m.version(p)}' for p in ('sglang-kernel','flashinfer-python','transformers','gguf')),flush=True)
PY
if [[ -d /ee-phase ]]; then
  export PYTHONPATH="${PYTHONPATH:-}:/ee-input"
fi
exec bash /tmp/ee-v3-service.sh
