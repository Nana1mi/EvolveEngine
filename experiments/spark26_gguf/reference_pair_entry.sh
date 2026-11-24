#!/usr/bin/env bash
# Paired-diagnostic timing: exactly the baked launcher; no shadow, audit or profiler.
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[ab][1-4]$ ]]
label=$1
src=/patched/python/sglang/srt/layers/quantization/gguf.py
if [[ "$label" == b* ]]; then
    script=$(python -c 'import json; print(json.load(open("/diagnostic/calibration_workload.json")).get("candidate_script", "reference_coalesce.py"))')
    [[ "$script" == reference_qkv.py || "$script" == reference_coalesce.py ]]
    python "/diagnostic/$script" /patched >&2
fi
actual=$(sha256sum "$src" | cut -d' ' -f1)
expected=$(python -c 'import sys,json; print(json.load(open("/diagnostic/calibration_workload.json"))["source_sha256"][sys.argv[1]])' "${label:0:1}")
[[ "$actual" == "$expected" ]]
printf 'EE_REFERENCE_PAIR %s %s\n' "$label" "$actual" >&2
bash /opt/spark26/e0_entry.sh server >&2 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
SECONDS=0
while (( SECONDS < 240 )); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; exit 1; fi
    if curl -fsS --max-time 3 http://127.0.0.1:30091/health >/dev/null 2>&1; then
        python -u /diagnostic/synthetic_baseline.py "$label"
        exit
    fi
    sleep 1
done
echo HEALTH_TIMEOUT >&2
exit 1
