#!/usr/bin/env bash
# Diagnostic-only candidate shadow semantics & residence, no timing claims.
set -euo pipefail
[[ -d /patched/python && -r /diagnostic/reference_coalesce.py ]]
python /diagnostic/reference_coalesce.py /patched --verify-exact
exec bash /diagnostic/runtime_residency_entry.sh
