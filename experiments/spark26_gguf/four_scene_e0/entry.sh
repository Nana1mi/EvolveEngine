#!/usr/bin/env bash
set -euo pipefail
cd /diag
printf '%s  %s\n' "$(python -c 'import json;print(json.load(open("plan.json"))["gguf_py"])')" /patched/python/sglang/srt/layers/quantization/gguf.py | sha256sum -c - >&2
python - <<'PY'
from pathlib import Path
import json
p=json.loads(Path('plan.json').read_text()); s=Path('/opt/spark26/e0_entry.sh').read_text()
for flag,old,new in [('context-length',1024,p['context_length']),('max-total-tokens',4096,p['max_total_tokens']),('max-running-requests',2,p['max_running_requests'])]:
    a=f'--{flag} {old} '; assert s.count(a)==1; s=s.replace(a,f'--{flag} {new} ')
Path('/tmp/ee-server.sh').write_text(s)
PY
bash /tmp/ee-server.sh server >/tmp/ee-e0-server.log 2>&1 & pid=$!
cleanup(){ rc=$?; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; cat /tmp/ee-e0-server.log >&2; exit "$rc"; }; trap cleanup EXIT
for ((i=0;i<240;i++)); do
  kill -0 "$pid" || exit 4
  if curl -fsS --max-time 2 http://127.0.0.1:30091/health >/dev/null 2>&1; then
    python -u /diag/probe.py "$1"; exit
  fi
  sleep 1
done
exit 5
