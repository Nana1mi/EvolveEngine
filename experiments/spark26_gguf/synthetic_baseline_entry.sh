#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" =~ ^([ab][123]|profile)$ ]]
# Use a per-container key for the authenticated profiler only; timed containers
# do not enable its admin API. There is no host port and no external network.
if [[ "$1" == profile ]]; then
    export SGLANG_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
fi
bash /opt/spark26/e0_entry.sh server >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 3 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python -u /diagnostic/synthetic_baseline.py "$1"
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
