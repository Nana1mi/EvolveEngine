#!/usr/bin/env bash
# Candidate-specific deterministic gate; no timing measurements or live deployment.
set -euo pipefail
[[ $# == 1 && "$1" =~ ^det[12]$ ]]
python /diagnostic/reference_qkv.py /patched >&2
source=/patched/python/sglang/srt/layers/quantization/gguf.py
actual=$(sha256sum "$source" | cut -d' ' -f1)
expected=$(python -c 'import json; print(json.load(open("/diagnostic/qkv_service_workload.json"))["source_sha256"]["b"])')
[[ "$actual" == "$expected" ]]
printf 'EE_REFERENCE_QKV_QUALITY %s %s\n' "$1" "$actual" >&2
exec bash /diagnostic/deterministic_probe_entry.sh "$1"
