#!/usr/bin/env bash
# Only inside a disposable E0 container; edits the container's private code copy.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:/diagnostic"
python /diagnostic/runtime_residency_probe.py
bash /opt/spark26/e0_entry.sh server >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 3 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        PYTHONUNBUFFERED=1 python - <<'PY'
import json
from pathlib import Path
from synthetic_baseline import emit, wave
from runtime_residency_probe import STAMP
shapes = json.loads(Path('/diagnostic/calibration_workload.json').read_bytes())['shapes']
STAMP.write_text('ready', encoding='utf-8')
wave('ee-resident-ready', shapes['short1'])
for name, spec in shapes.items():
    result = wave('ee-resident-'+name, spec)
    emit(kind='residency_workload', shape=name, outputs=[row['output_sha256'] for row in result['requests']])
STAMP.write_text('post_requests', encoding='utf-8')
wave('ee-resident-post', shapes['short1'])
assert not STAMP.exists(), 'Snapshot did not execute'
PY
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
