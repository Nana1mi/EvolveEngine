#!/usr/bin/env bash
set -Eeuo pipefail; umask 077
[[ $(hostname) == spark-c3bc && $(id -un) == jiazheng ]]; root=$(realpath "$1"); [[ $root == "$HOME"/spark26-v32-g1-v.* && -O $root ]]; cd "$root"; [[ ! -e run.closed && ! -e run-order.tsv ]]; exec 9>"$HOME/.evolve-spark26-gpu.lock"; flock -n 9
printf '%s  bundle.sha256\n' "$2" | sha256sum -c -; sha256sum -c bundle.sha256
IFS= read -r pw; sudo -S -p '' -v <<<"$pw"; unset pw
c=''; cleanup(){ rc=$?; if [[ -n $c ]]; then sudo -n docker rm -f "$c" >/dev/null 2>&1 || true; fi; nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv >post-gpu.csv; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader >post-compute.csv; printf '%s\t%s\n' "$(TZ=Asia/Hong_Kong date -Is)" "$rc" >run.closed; sudo -k; exit "$rc"; }; trap cleanup EXIT; trap 'exit 143' TERM HUP
base="$HOME/spark26-q8-e0.X4xmPm"; model="$HOME/spark26-q8-smoke-20261122.q8R55w/model"; weight=$(cat "$base/weight-path.txt"); image=$(cat "$base/image-id.txt"); [[ $image == $(python3 -c 'import json;print(json.load(open("plan.json"))["image"])') && $(sudo -n docker image inspect --format '{{.Id}}' "$image") == "$image" ]]; (cd "$model"; sha256sum -c "$root/model.sha256") >model.verify; echo "$(python3 -c 'import json;print(json.load(open("plan.json"))["weight"])')  $weight" | sudo -n sha256sum -c - >>model.verify
for l in e0a e0b exact fp64greedy faultv; do
  sudo -n -v; [[ -z $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) ]] || { echo occupied >&2; exit 6; }
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv >"$l.pre.csv"
  printf 'start\t%s\t%s\t%s\n' "$l" "$(date +%s)" "$(TZ=Asia/Hong_Kong date -Is)" | tee -a run-order.tsv
  c="ee-g1v-${root##*.}-$l"; c=${c,,}; if timeout --kill-after=15s 360s sudo -n docker run --rm --name "$c" --network none --gpus all --shm-size 2g --mount "type=bind,src=$model,dst=/model,readonly" --mount "type=bind,src=$weight,dst=/model/Spark-X2.6-1.7B-Q8_0.gguf,readonly" --mount "type=bind,src=$root,dst=/diag,readonly" --entrypoint bash "$image" /diag/entry.sh "$l" >"$l.jsonl" 2>"$l.server.log"; then c=''; else grep -Ei 'error|exception|traceback|unverified|failed' "$l.server.log" | tail -10 | cut -b 1-450 || true; exit 7; fi
  expected=32; if [[ $l =~ ^(e0a|e0b|exact)$ ]]; then expected=35; fi; [[ $(wc -l <"$l.jsonl") == $expected ]]; sha256sum "$l.jsonl" "$l.server.log" "$l.pre.csv" >>raw.sha256
  printf 'done\t%s\t%s\t%s\n' "$l" "$(date +%s)" "$(TZ=Asia/Hong_Kong date -Is)" | tee -a run-order.tsv
done
sha256sum -c raw.sha256 >/dev/null
