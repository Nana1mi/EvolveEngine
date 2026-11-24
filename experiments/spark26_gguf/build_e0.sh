#!/usr/bin/env bash
# Context directory is prepared with the exact source archive and hashed wheel inputs.
set -euo pipefail
if (( $# != 3 )); then echo "Usage: $0 <context-dir> <local-base-tag> <output-tag>" >&2; exit 2; fi
context="$1"; base="$2"; tag="$3"
base_id=sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701
actual=$(docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$base")
[[ "$actual" == "$base_id linux/arm64" ]] || { echo "Unexpected base: $actual" >&2; exit 2; }
# Reject files outside the locked wheel list before pip's glob can consume them.
(
    cd "$context"
    sha256sum -c SHA256SUMS
    locked=$(awk '$2 ~ /^wheels\// { print $2 }' SHA256SUMS | LC_ALL=C sort)
    present=$(printf '%s\n' wheels/*.whl | LC_ALL=C sort)
    [[ "$present" == "$locked" ]] || { echo 'Wheel list differs from the lock' >&2; exit 2; }
)
# Use the node-local base. No dependency resolver, network install, or mutable remote base pull.
DOCKER_BUILDKIT=0 docker build --pull=false --network=none \
    --build-arg "BASE_IMAGE=$base" --build-arg "BASE_ID=$base_id" \
    --file "$context/Dockerfile.e0" --tag "$tag" "$context"
[[ "$(docker image inspect --format '{{index .Config.Labels "ai.evolveengine.base-id"}}' "$tag")" == "$base_id" ]]
