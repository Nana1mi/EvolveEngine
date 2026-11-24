#!/usr/bin/env bash
# .108 only: receives sudo input through the authenticated local task helper.
set -euo pipefail
umask 077
[[ $# == 2 && "$(hostname)" == spark-c3bc && "$(id -un)" == jiazheng ]]
root=$(realpath "$1")
[[ -O "$root" && -d "$root" ]]
case "$root" in
    "$HOME"/spark26-q8-reference-pairs.*) workload=reference_pair_workload.json; candidate=reference_coalesce.py ;;
    "$HOME"/spark26-q8-qkv-service.*) workload=qkv_service_workload.json; candidate=reference_qkv.py ;;
    *) exit 2 ;;
esac
[[ "$2" =~ ^[0-9a-f]{64}$ ]]
cd "$root"
exec 9>"$HOME/.evolve-spark26-gpu.lock"
flock -n 9
[[ ! -e run-order.tsv && ! -e run.closed ]]
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
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$workload")
want_weight=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gguf_sha256"])' "$workload")
[[ $(cat "$prior/image-id.txt") == "$image_id" && $(sudo -n docker image inspect --format '{{.Id}}' "$image_id") == "$image_id" ]]
(cd "$model"; sha256sum -c "$prior/model-files.sha256") > model.verify
[[ $(cut -d' ' -f1 "$prior/weight.sha256") == "$want_weight" ]]
sudo -n sha256sum -c "$prior/weight.sha256" >> model.verify
expected=$(python3 -c 'import json,sys; print(" ".join(l for p in json.load(open(sys.argv[1]))["pairs"] for l in p["labels"]))' "$workload")
[[ "$expected" == 'a1 b1 b2 a2 b3 a3 a4 b4' ]]
fields='name,driver_version,temperature.gpu,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw'
for label in $expected; do
    processes=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    [[ -z "$processes" ]] || { printf 'GPU occupied:\n%s\n' "$processes" >&2; exit 3; }
    nvidia-smi --query-gpu="$fields" --format=csv > "$label.pre-gpu.csv"
    suffix=${root##*.}; suffix=${suffix,,}
    container="ee-spark26-ref-${suffix}-${label}"
    printf 'start\t%s\t%s\n' "$label" "$(TZ=Asia/Hong_Kong date -Is)" | tee -a run-order.tsv
    if ! timeout --kill-after=15s 420s sudo -n docker run --rm --name "$container" \
        --network none --gpus all --shm-size 2g \
        --mount "type=bind,src=$model,dst=/model,readonly" \
        --mount "type=bind,src=$weight,dst=/model/$weightname,readonly" \
        --mount "type=bind,src=$root/$workload,dst=/diagnostic/calibration_workload.json,readonly" \
        --mount "type=bind,src=$root/$candidate,dst=/diagnostic/$candidate,readonly" \
        --mount "type=bind,src=$root/reference_pair_entry.sh,dst=/diagnostic/reference_pair_entry.sh,readonly" \
        --mount "type=bind,src=$root/synthetic_baseline.py,dst=/diagnostic/synthetic_baseline.py,readonly" \
        --entrypoint bash "$image_id" /diagnostic/reference_pair_entry.sh "$label" \
        > "$label.stdout" 2> "$label.server.log"; then
        tail -25 "$label.server.log" | cut -b 1-420 >&2; exit 4
    fi
    container=''
    [[ $(grep -c '"kind": "wave"' "$label.stdout") == 28 ]]
    nvidia-smi --query-gpu="$fields" --format=csv > "$label.post-gpu.csv"
    sha256sum "$label.stdout" "$label.server.log" "$label.pre-gpu.csv" "$label.post-gpu.csv" >> raw-evidence.sha256
    printf 'done\t%s\t%s\n' "$label" "$(TZ=Asia/Hong_Kong date -Is)" | tee -a run-order.tsv
done
sha256sum -c raw-evidence.sha256 >/dev/null
