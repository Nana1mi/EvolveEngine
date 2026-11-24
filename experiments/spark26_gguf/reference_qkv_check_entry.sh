#!/usr/bin/env bash
# Separate QKV-only bit-pattern oracle and residence diagnostic, not a benchmark.
set -euo pipefail
python /diagnostic/reference_qkv.py /patched --verify-exact
exec bash /diagnostic/runtime_residency_entry.sh
