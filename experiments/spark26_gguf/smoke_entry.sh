#!/usr/bin/env bash
# Run only in a disposable, network-isolated container with the declared mounts.
set -euo pipefail
python -c 'import sglang, torch; assert sglang.__file__.startswith("/patched/python/"); assert torch.cuda.is_available(); print(sglang.__file__, torch.__version__, torch.cuda.get_device_name())'
python -m sglang.launch_server \
    --model-path /model/Spark-X2.6-1.7B-Q8_0.gguf --load-format gguf \
    --tokenizer-path /model --chat-template /model/chat_template.jinja --trust-remote-code \
    --host 127.0.0.1 --port 30091 --served-model-name spark26-q8 --tp-size 1 \
    --context-length 1024 --max-total-tokens 4096 --mem-fraction-static 0.4 \
    --max-running-requests 2 --attention-backend triton \
    --disable-decode-cuda-graph --disable-prefill-cuda-graph \
    --tool-call-parser spark25 --reasoning-parser qwen3 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 4 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python - <<'PY'
import json
from urllib.request import Request, build_opener, ProxyHandler

body = dict(model='spark26-q8', messages=[dict(role='user', content='Reply with the single word READY.')],
            chat_template_kwargs=dict(enable_thinking=False), temperature=0, max_tokens=16, stream=True)
request = Request('http://127.0.0.1:30091/v1/chat/completions', json.dumps(body).encode(),
                  {'Content-Type': 'application/json'})
chunks, finish, done = [], None, False
with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
    if response.headers.get_content_type() != 'text/event-stream':
        raise ValueError('Expected a streaming response')
    for line in response:
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            done = True
            break
        event = json.loads(data)
        if 'error' in event:
            raise ValueError(event['error'])
        for choice in event.get('choices', ()):
            delta = choice.get('delta', {}).get('content')
            if delta:
                chunks.append(delta)
            finish = choice.get('finish_reason') or finish
text = ''.join(chunks)
if not (done and finish == 'stop' and text.strip().upper().rstrip('.') == 'READY'):
    raise ValueError(f'Incomplete or unexpected reply: {text!r} (finish={finish}, done={done})')
print(json.dumps(dict(text=text, content_chunks=len(chunks), finish=finish, done=done)))
PY
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
