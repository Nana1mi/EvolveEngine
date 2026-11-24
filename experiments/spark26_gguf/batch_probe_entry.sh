#!/usr/bin/env bash
# Read-only mounted diagnostics. Derive the server command from the exact baked launcher.
set -euo pipefail
source_entry=/opt/spark26/e0_entry.sh
needle='--tool-call-parser spark25 --reasoning-parser qwen3)'
[[ $(grep -Fc -- "$needle" "$source_entry") == 1 ]]
sed 's/--tool-call-parser spark25 --reasoning-parser qwen3)/--tool-call-parser spark25 --reasoning-parser qwen3 --enable-metrics --log-schedule-decisions)/' \
    "$source_entry" > /tmp/batch-probe-engine.sh
bash /tmp/batch-probe-engine.sh server >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 3 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python -u /diagnostic/batch_probe.py
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
