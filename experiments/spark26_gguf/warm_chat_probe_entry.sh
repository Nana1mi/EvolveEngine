#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" =~ ^(control|primed)$ ]]
# Use the exact immutable E0 image startup; no extra metrics or tracing flags.
bash /opt/spark26/e0_entry.sh server >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 3 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python -u /diagnostic/warm_chat_probe.py "$1"
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
