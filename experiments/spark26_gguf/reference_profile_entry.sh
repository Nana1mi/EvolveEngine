#!/usr/bin/env bash
# Shape-aware tracing for the rejected candidate. Never a performance sample.
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[ab]$ ]]
if [[ "$1" == b ]]; then
    python /diagnostic/reference_coalesce.py /patched >&2
fi
src=/patched/python/sglang/srt/layers/quantization/gguf.py
actual=$(sha256sum "$src" | cut -d' ' -f1)
expected=$(python -c 'import json,sys; print(json.load(open("/diagnostic/reference_pair_workload.json"))["source_sha256"][sys.argv[1]])' "$1")
[[ "$actual" == "$expected" ]]
printf 'EE_REFERENCE_PROFILE %s %s\n' "$1" "$actual" >&2
export EE_SPARK26_PROFILE_SHAPES=1
exec bash /diagnostic/synthetic_baseline_entry.sh profile
