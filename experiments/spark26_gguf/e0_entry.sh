#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in server|quality|bench) mode="$1"; shift;; *) echo 'Expected server, quality, or bench' >&2; exit 2;; esac
if (( $# )); then [[ "$mode" == bench && $# == 1 && "$1" == reverse ]] || exit 2; fi
python - <<'PY'
import importlib.metadata as m
import sglang
import torch
assert sglang.__file__.startswith('/patched/python/'), sglang.__file__
assert torch.cuda.is_available()
print('runtime', sglang.__file__, torch.__version__, torch.cuda.get_device_name(),
      *(f'{p}={m.version(p)}' for p in ('sglang-kernel','flashinfer-python','transformers','gguf')), flush=True)
PY
args=(--model-path /model/Spark-X2.6-1.7B-Q8_0.gguf --load-format gguf
      --tokenizer-path /model --chat-template /model/chat_template.jinja --trust-remote-code
      --host 127.0.0.1 --port 30091 --served-model-name spark26-q8 --tp-size 1
      --context-length 1024 --max-total-tokens 4096 --mem-fraction-static 0.4
      --max-running-requests 2 --attention-backend triton --disable-radix-cache
      --stream-interval 1 --disable-decode-cuda-graph --disable-prefill-cuda-graph
      --tool-call-parser spark25 --reasoning-parser qwen3)
if [[ "$mode" == server ]]; then exec python -m sglang.launch_server "${args[@]}"; fi
python -m sglang.launch_server "${args[@]}" >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 4 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python -u /opt/spark26/e0_probe.py "$mode" "$@"
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
