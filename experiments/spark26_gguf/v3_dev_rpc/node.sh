#!/usr/bin/env bash
set -Eeuo pipefail; umask 077
[[ $(hostname) == spark-c3bc && $(id -un) == jiazheng ]]; root=$(realpath "$1"); [[ $root == "$HOME"/spark26-v3-rpc.* && -O $root ]]; cd "$root"
[[ ! -e run.closed && ! -e probe.jsonl ]]; exec 9>"$HOME/.evolve-spark26-gpu.lock"; flock -n 9
printf '%s  bundle.sha256\n' "$2" | sha256sum -c -; sha256sum -c bundle.sha256
printf '%s  service/patch.json\n' "$3" | sha256sum -c -
printf '%s  service/candidate-gguf.py\n' "$4" | sha256sum -c -
IFS= read -r pw; sudo -S -p '' -v <<<"$pw"; unset pw
base="$HOME/spark26-q8-e0.X4xmPm"; model="$HOME/spark26-q8-smoke-20261122.q8R55w/model"; weight=$(cat "$base/weight-path.txt"); image=$(cat "$base/image-id.txt")
svc=''; client=''; copied=''; started=$(date +%s%N)
cleanup(){
  rc=$?; if [[ -n $svc ]]; then sudo -n docker inspect "$svc" --format '{{.Image}} {{.HostConfig.NetworkMode}} {{json .Mounts}}' >service.inspect 2>/dev/null || true; sudo -n docker logs "$svc" >service.log 2>&1 || true; fi
  if [[ -n $client ]]; then sudo -n docker rm -f "$client" >/dev/null 2>&1 || true; fi
  if [[ -n $svc ]]; then sudo -n docker rm -f "$svc" >/dev/null 2>&1 || true; fi
  if [[ -n $copied && $copied == "$root/isolated-weight.gguf" ]]; then rm -f -- "$copied"; fi
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader >post-compute.csv
  printf '%s\t%s\t%s\n' "$started" "$(date +%s%N)" "$rc" >run.closed; sudo -k; exit "$rc"
}
trap cleanup EXIT; trap 'exit 143' TERM HUP
readarray -t values < <(python3 - <<'PY'
import json
p=json.load(open('service/plan.json'));s=json.load(open('service/patch.json'));print(p['image']);print(p['weight']); print(s['operation']); print(s['node_limit_ms'])
PY
)
[[ $image == "${values[0]}" && $(sudo -n docker image inspect --format '{{.Id}}' "$image") == "$image" && ${values[2]} =~ ^develop_(quality|measure)$ && ${values[3]} =~ ^[1-9][0-9]{2,5}$ ]]
limit=$((${values[3]}/1000)); [[ $limit -ge 50 && $limit -le 360 ]]
(cd "$model"; sha256sum -c "$root/model.sha256") >model.verify; echo "${values[1]}  $weight" | sudo -n sha256sum -c - >>model.verify
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo occupied >&2; exit 6; }
runtimeWeight="$weight"
if [[ ! -r $weight ]]; then
  required=$(sudo -n stat -c %s -- "$weight"); available=$(df -PB1 "$root" | awk 'NR==2 {print $4}')
  [[ $required =~ ^[0-9]+$ && $available =~ ^[0-9]+$ && $available -gt $((required+536870912)) ]]
  copied="$root/isolated-weight.gguf"; [[ ! -e $copied ]]
  sudo -n install -m 0400 -o "$(id -u)" -g "$(id -g)" -- "$weight" "$copied"
  echo "${values[1]}  $copied" | sha256sum -c - >staged-weight.verify
  runtimeWeight="$copied"
fi
suffix=${root##*.}; suffix=${suffix,,}; svc="ee-v3-rpc-$suffix-engine"; client="ee-v3-rpc-$suffix-observer"
phase_service=(); phase_observer=()
if [[ ${values[2]} == develop_quality ]]; then
  mkdir -m 0700 -- "$root/phase" "$root/instrumented"
  sudo -n docker run --rm --network none --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    --mount "type=bind,src=$root/service,dst=/ee-input,readonly" --mount "type=bind,src=$root/instrumented,dst=/output" \
    --entrypoint python "$image" -B /ee-input/residency_hook.py
  phase_service=(--mount "type=bind,src=$root/phase,dst=/ee-phase,readonly"
    --mount "type=bind,src=$root/instrumented/model_runner.py,dst=/patched/python/sglang/srt/model_executor/model_runner.py,readonly"
    --mount "type=bind,src=$root/instrumented/spark2_5.py,dst=/patched/python/sglang/srt/models/spark2_5.py,readonly")
  phase_observer=(--mount "type=bind,src=$root/phase,dst=/ee-phase")
fi
sudo -n docker run -d "${phase_service[@]}" --name "$svc" --network none --user "$(id -u):$(id -g)" --env HOME=/tmp --gpus all --cap-drop ALL --security-opt no-new-privileges --shm-size 2g --mount "type=bind,src=$model,dst=/model,readonly" --mount "type=bind,src=$runtimeWeight,dst=/model/Spark-X2.6-1.7B-Q8_0.gguf,readonly" --mount "type=bind,src=$root/service,dst=/ee-input,readonly" --mount "type=bind,src=$root/service/candidate-gguf.py,dst=/patched/python/sglang/srt/layers/quantization/gguf.py,readonly" --entrypoint bash "$image" /ee-input/service.sh >service.id
sudo -n docker create "${phase_observer[@]}" --name "$client" --network "container:$svc" --user "$(id -u):$(id -g)" --env HOME=/tmp --cap-drop ALL --security-opt no-new-privileges --mount "type=bind,src=$root/observer,dst=/dev-probe,readonly" --mount "type=bind,src=$model,dst=/model,readonly" --entrypoint python "$image" -B -u /dev-probe/probe.py "${values[2]}" >observer.id
sudo -n docker inspect "$client" --format '{{.Image}} {{.HostConfig.NetworkMode}} {{json .Mounts}}' >observer.inspect
spent=$(( ($(date +%s%N)-started)/1000000000 )); remaining=$((limit-spent-12)); [[ $remaining -ge 30 ]]
timeout --kill-after=5s "${remaining}s" sudo -n docker start -a "$client" >probe.jsonl 2>probe.stderr & probe_pid=$!
while kill -0 "$probe_pid" 2>/dev/null; do
  if [[ $(sudo -n docker inspect --format '{{.State.Running}}' "$svc") != true ]]; then
    sudo -n docker stop --time 1 "$client" >/dev/null 2>&1 || true; wait "$probe_pid" || true; exit 7
  fi
  sleep 1
done
wait "$probe_pid"
sha256sum probe.jsonl >raw.sha256
