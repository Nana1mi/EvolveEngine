#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]]
export EE_SPARK26_ALLOCATOR_TRACE=1
exec bash /diagnostic/runtime_residency_entry.sh
