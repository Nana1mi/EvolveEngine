#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 2 && "$(hostname)" == spark-c3bc && "$(id -un)" == jiazheng ]]; [[ "$2" =~ ^[0-9a-f]{64}$ ]]
root=$(realpath "$1")
[[ "$root" == "$HOME"/spark26-q8-qkv-micro.* && -O "$root" && -d "$root" ]]
cd "$root"
exec 9>"$HOME/.evolve-spark26-gpu.lock"
flock -n 9
[[ ! -e micro.stdout && ! -e run.closed ]]
printf '%s  bundle.sha256\n' "$2" | sha256sum -c -
sha256sum -c bundle.sha256
IFS= read -r pw
sudo -S -p '' -v <<< "$pw"
unset pw
container=''
cleanup() {
    local rc=$?
    if [[ -n "$container" ]]; then sudo -n docker rm -f "$container" >/dev/null 2>&1 || true; fi
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv > post-gpu.csv || true
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > post-compute.csv || true
    printf '%s\t%s\n' "$(TZ=Asia/Hong_Kong date -Is)" "$rc" > run.closed
    sudo -k
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
prior="$HOME/spark26-q8-e0.X4xmPm"
model="$HOME/spark26-q8-smoke-20261122.q8R55w/model"
weightname=Spark-X2.6-1.7B-Q8_0.gguf
weight=$(cat "$prior/weight-path.txt")
image_id=$(python3 -c 'import json; print(json.load(open("qkv_micro_lock.json"))["image_id"])')
want_weight=$(python3 -c 'import json; print(json.load(open("qkv_micro_lock.json"))["gguf_sha256"])')
[[ $(cat "$prior/image-id.txt") == "$image_id" && $(sudo -n docker image inspect --format '{{.Id}}' "$image_id") == "$image_id" ]]
(cd "$model"; sha256sum -c "$prior/model-files.sha256") > model.verify
[[ $(cut -d' ' -f1 "$prior/weight.sha256") == "$want_weight" ]]
sudo -n sha256sum -c "$prior/weight.sha256" >> model.verify
processes=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[[ -z "$processes" ]] || { printf 'GPU occupied:\n%s\n' "$processes" >&2; exit 3; }
suffix=${root##*.}; suffix=${suffix,,}; container="ee-spark26-qkv-micro-${suffix}"
printf 'Start %s\n' "$(TZ=Asia/Hong_Kong date -Is)"
if ! timeout --kill-after=12s 300s sudo -n docker run --rm --name "$container" \
    --network none --gpus all --shm-size 2g \
    --mount "type=bind,src=$model,dst=/model,readonly" \
    --mount "type=bind,src=$weight,dst=/model/$weightname,readonly" \
    --mount "type=bind,src=$root,dst=/diagnostic,readonly" \
    --entrypoint python "$image_id" -u /diagnostic/qkv_micro.py > micro.stdout 2> micro.stderr; then
    tail -35 micro.stderr | cut -b 1-420 >&2
    exit 4
fi
container=''
grep -q '"kind": "complete"' micro.stdout
sha256sum micro.stdout micro.stderr > raw-evidence.sha256
printf 'Done %s\n' "$(TZ=Asia/Hong_Kong date -Is)"
