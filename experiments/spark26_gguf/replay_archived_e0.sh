#!/usr/bin/env bash
# Independently verify the archived E0 on an authorized, idle ARM64/GB10 node.
# Docker access is provided by the caller; no persistent service is created.
set -euo pipefail
umask 077
if (( $# != 4 )); then
    echo "Usage: $0 <docker.tar.zst> <model-config-dir> <q8.gguf> <new-evidence-dir>" >&2
    exit 2
fi
repo=$(dirname "$(realpath "$0")")
archive=$(realpath -e "$1"); model=$(realpath -e "$2"); weight=$(realpath -e "$3")
[[ -f "$archive" && -d "$model" && -f "$weight" ]]
expected='sha256:28e47ed30260eaaab0fee0251b53018766afc2dc61e1c8e4c65be4b1e6d088ff'
expected_tag='evolveengine/spark26-q8-diagnostic-e0:x4xmpm'
archive_sha='ae4f4a6de95c4027f2cbd420732b974e13e92a333d8733be35df8fd7135fcf94'
weight_sha='d1df86f9332b75424ab3f7e14fdc2cf20aec93fa1e4a86393bb4e75057f24180'
[[ "$(uname -m)" == aarch64 ]]
mkdir -- "$4"
out=$(realpath -e "$4")
assert_idle() {
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    [[ -z "$procs" ]] || { printf 'Refusing an occupied node:\n%s\n' "$procs" >&2; return 3; }
}
assert_idle
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu --format=csv > "$out/pre-gpu.csv"
grep -q 'NVIDIA GB10' "$out/pre-gpu.csv"
{
    hostname; uname -m; TZ=Asia/Hong_Kong date -Is
    docker version --format '{{.Server.Version}}'
} > "$out/node.txt"
(d=$(sha256sum -- "$archive"); printf '%s\n' "$d"; [[ "${d%% *}" == "$archive_sha" ]]) > "$out/archive.verify"
(d=$(sha256sum -- "$weight"); printf '%s\n' "$d"; [[ "${d%% *}" == "$weight_sha" ]]) > "$out/weight.verify"
(cd "$model" && sha256sum -c "$repo/diagnostic-e0-model.sha256") > "$out/model.verify"
# The optional file bind below assumes a regular target file in the staging directory.
# Avoid Docker creating a mount target inside a model directory or following a symlink.
weightname=Spark-X2.6-1.7B-Q8_0.gguf
[[ -f "$model/$weightname" && ! -L "$model/$weightname" ]]
if docker image inspect "$expected_tag" >/dev/null 2>&1; then
    [[ "$(docker image inspect --format '{{.Id}}' "$expected_tag")" == "$expected" ]]
fi
if docker image inspect "$expected" >/dev/null 2>&1; then
    echo 'present_before_load' > "$out/preexisting-image.txt"
else
    echo 'absent_before_load' > "$out/preexisting-image.txt"
fi
zstd -dc -- "$archive" | docker image load > "$out/docker-load.log" 2>&1
loaded=$(docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$expected_tag")
[[ "$loaded" == "$expected linux/arm64" ]]
printf '%s\n' "$loaded" > "$out/image-id.txt"
assert_idle
name="ee-spark26-portability-$(date +%s)-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
timeout --kill-after=15s 410s docker run --rm --name "$name" --network none --gpus all --shm-size 2g \
    --mount "type=bind,src=$model,dst=/model,readonly" \
    --mount "type=bind,src=$weight,dst=/model/$weightname,readonly" \
    "$expected" quality > "$out/quality.stdout" 2> "$out/quality.server.log"
grep -F '"kind": "quality_summary", "passed": 6, "total": 6' "$out/quality.stdout"
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu --format=csv > "$out/post-gpu.csv"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/post-compute.csv"
