"""DGX Spark host worker invoked locally or over SSH by the controller."""

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import zlib

from .schema import canonical, digest
from .spark_runtime import WIRE
from .workspace import Repository


def asset_digest(root):
    if not root.is_dir():
        raise ValueError("Model path must be a directory")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("The pinned model asset directory is empty")
    for path in files:
        item = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(2**20), b""):
                item.update(block)
        digest.update(canonical([path.relative_to(root).as_posix(), item.hexdigest()]).encode())
    return digest.hexdigest()


class Container:
    def __init__(self, config, tree, deadline, reference=False):
        self.config, self.tree, self.deadline, self.reference = config, tree, deadline, reference
        self.name = "evolveengine-" + uuid.uuid4().hex
        self.lines = queue.Queue(maxsize=2)

    def __enter__(self):
        config = self.config
        package = Path(__file__).resolve().parent
        model = Path(config["model_path"]).resolve(strict=True)
        image = config["image"]
        if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image):
            raise ValueError("GPU worker image must be pinned by OCI digest")
        args = ["docker", "run", "--pull=never", "--rm", "--name", self.name, "-i",
                "--network", "none", "--gpus", "device=0", "--cap-drop", "ALL", "--read-only",
                "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:exec,mode=1777",
                "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
                "-e", "PYTHONPATH=/app:/engine/python",
                "-e", "TORCH_CUDA_ARCH_LIST=12.1",
                "-v", f"{package}:/app/evolveengine:ro", "-v", f"{model}:/model:ro"]
        if not self.reference:
            args += ["-v", f"{self.tree}:/engine:ro"]
        for key, value in config.get("runtime_env", {}).items():
            args += ["-e", f"{key}={value}"]
        args += ["--workdir", "/tmp", "--entrypoint", "python3", image, "-m", "evolveengine.spark_runtime"]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.send({"reference": self.reference, "engine_options": config["engine_options"]})
            self.info = self.receive()["ready"]
        except BaseException:
            self.__exit__()
            raise
        return self

    def _read(self):
        for line in self.process.stdout:
            if line.startswith(WIRE):
                self.lines.put(line[len(WIRE):])
        self.lines.put(None)

    def send(self, value):
        self.process.stdin.write(canonical(value) + "\n")
        self.process.stdin.flush()

    def receive(self):
        try:
            reply = self.lines.get(timeout=max(0, self.deadline - time.monotonic()))
        except queue.Empty as failure:
            raise TimeoutError("GPU worker exceeded its operation deadline") from failure
        if reply is None:
            raise RuntimeError("GPU worker exited before returning a receipt")
        return json.loads(reply)

    def token(self, prompt, count, ignore_eos=True, **metadata):
        self.send({"profile": metadata.pop("profile", False),
                   "input": {"prompt": prompt, "max_new_tokens": count, "ignore_eos": ignore_eos, **metadata}})
        return self.receive()

    def __exit__(self, *_):
        try:
            if self.process.poll() is None:
                self.send({"shutdown": True})
                self.process.wait(timeout=min(5, max(0, self.deadline - time.monotonic())))
        except (OSError, subprocess.TimeoutExpired):
            try:
                subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=5)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.process.stdin.close()
            self.process.stdout.close()


def sensor():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.gr,clocks.mem,memory.used",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10,
    )
    keys = ("uuid", "model", "driver", "temperature_c", "graphics_mhz", "memory_mhz", "memory_mib")
    records = [dict(zip(keys, (column.strip() for column in row.split(","))))
               for row in result.stdout.strip().splitlines()]
    if len(records) != 1 or "GB10" not in records[0]["model"]:
        raise ValueError("GPU inventory is not a single GB10")
    return records[0]


@contextmanager
def exclusive(path, deadline):
    import fcntl

    with Path(path).open("a+b") as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as failure:
                if time.monotonic() + 0.1 >= deadline:
                    raise TimeoutError("Spark node lease unavailable") from failure
                time.sleep(0.1)
        yield


def measure(worker, block):
    limit = block["max_new_tokens"]
    prompts = block["prompts"]
    for prompt in prompts[:4]:
        result = worker.token(prompt[::-1], limit)
        if result["output_tokens"] != limit or result["finish_reason"] != "length":
            raise ValueError("Warmup violated fixed-length serving")
    rows = []
    start = time.perf_counter()
    for prompt in prompts:
        request_at = time.perf_counter()
        result = worker.token(prompt, limit)
        result["request_at_seconds"] = request_at - start
        result["reply_at_seconds"] = time.perf_counter() - start
        rows.append(result)
    end = time.perf_counter()
    return {"elapsed_seconds": end - start, "requests": rows, "runtime": worker.info}


def profile(worker, config):
    inputs = config["profile_prompt"]
    if not inputs or not all(type(value) is int and 0 <= value < config["vocab_size"] for value in inputs):
        raise ValueError("A valid, fixed profile prompt is required")
    result = worker.token(inputs, 32, profile=True)
    spans = sorted((float(event["ts"]), float(event["ts"]) + float(event["dur"]))
                   for event in result["trace"])
    measurement = result["measurement"]
    fraction = None
    if spans:
        first, end = spans[0]
        total = 0.0
        begin = first
        for left, right in spans:
            if left > end:
                total += end - begin
                begin = left
            end = max(right, end)
        total += end - begin
        if end > first:
            fraction = max(0.0, 1.0 - total / (end - first))
    gap = "unknown" if fraction is None else ("high" if fraction >= config["gap_threshold"] else "low")
    return {"observation": {"primary_class": "host_dispatch" if gap == "high" else "unknown",
                            "phase": "mixed", "host_launch_gap": gap},
            "evidence": {"kineto_gpu_activity_count": len(spans), "gpu_idle_fraction": fraction,
                         "time_interpretation": "prefill+decode; idle proxy, not CPU attribution",
                         "ttft_seconds": measurement["ttft_seconds"],
                         "tpot_seconds": measurement["tpot_seconds"]},
            "trace": result["trace"]}


def logit_checks(reference, candidate, vocab_size):
    import numpy as np

    if reference.keys() != candidate.keys() or not reference:
        return False, False, None
    maximum_kl = 0.0
    top = True
    full = True
    for position, encoded in reference.items():
        ref = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype="<f4")
        cand = np.frombuffer(zlib.decompress(base64.b64decode(candidate[position])), dtype="<f4")
        if (ref.size != vocab_size or cand.size != vocab_size or
                not np.isfinite(ref).all() or not np.isfinite(cand).all()):
            full = False
            continue
        i, j = ref.argmax(), cand.argmax()
        top &= (i == j or float(ref[i] - ref[j]) <= 0.05)
        left = ref.astype("float64") - float(ref.max())
        right = cand.astype("float64") - float(cand.max())
        left -= np.logaddexp.reduce(left)
        right -= np.logaddexp.reduce(right)
        maximum_kl = max(maximum_kl, float(np.dot(np.exp(left), left - right)))
    return bool(full), bool(top), maximum_kl


def quality(config, tree, block, deadline, role):
    # Independent model process writes reference results into the host evaluator's memory.
    count = config["quality_samples"]
    prompts = block["prompts"][:count]
    length = block["max_new_tokens"]
    positions = [item for item in (0, 1, 2, 16, 64, 127) if item < length]
    eos_prompts = config["eos_probes"][role]
    if type(count) is not int or not 1 <= count <= len(block["prompts"]) or not eos_prompts:
        raise ValueError("Quality gate needs a positive sample count and independent EOS probes")
    with Container(config, None, deadline, reference=True) as reference:
        expected = [reference.token(prompt, length, positions=positions) for prompt in prompts]
        termination = [reference.token(prompt, length, ignore_eos=False) for prompt in eos_prompts]
    checks = dict.fromkeys(("launch", "full_vocab", "kl", "top1", "sequence", "output_length", "eos", "kv_isolation"), True)
    checks["eos"] = any(row["finish_reason"] == "eos" and len(row["ids"]) < length for row in termination)
    records = []
    with Container(config, tree, deadline) as worker:
        for prompt, answer in zip(prompts, expected):
            observed = worker.token(prompt, length)
            forced = worker.token(prompt, length, positions=positions, forced=answer["ids"])
            full, top, kl = logit_checks(answer["logits"], forced["logits"], config["vocab_size"])
            checks["full_vocab"] &= full
            checks["kl"] &= kl is not None and kl <= config["max_kl"]
            checks["top1"] &= top
            checks["sequence"] &= observed["ids"] == answer["ids"]
            checks["output_length"] &= observed["output_tokens"] == length and observed["finish_reason"] == "length"
            records.append({"kl": kl, "top1": top, "tokens": observed["output_tokens"]})
        for prompt, answer in zip(eos_prompts, termination):
            observed = worker.token(prompt, length, ignore_eos=False)
            checks["eos"] &= observed["ids"] == answer["ids"] and observed["finish_reason"] == answer["finish_reason"]
        repeated = worker.token(prompts[0], length)
        checks["kv_isolation"] = repeated["ids"] == expected[0]["ids"]
        hardware = worker.info
    return {"receipt": {"passed": all(checks.values()), "checks": checks},
            "samples": records, "runtime": hardware}


def run(config, job):
    deadline = time.monotonic() + job["timeout_ms"] / 1000
    known = {"search", "development", "admission", "final", "preflight"}
    fixtures = config["eos_probes"]
    if set(fixtures) != known:
        raise ValueError("Register independent EOS fixtures for every evaluation role")
    eos_prompts = [tuple(prompt) for entries in fixtures.values() for prompt in entries]
    if len(eos_prompts) != len(set(eos_prompts)):
        raise ValueError("EOS fixtures must not be reused between roles")
    with exclusive(config["lease_path"], deadline):
        before = sensor()
        repository = Repository(config["engine_checkout"], config["engine_revision"], config["editable"])
        options = config["engine_options"]
        if options.get("use_dummy_weight") or options.get("dtype"):
            raise ValueError("Benchmark must use fixed real BF16 weights")
        with repository.checkout(job["engine"]) as tree:
            stage = job["stage"]
            if stage == "quality":
                result = quality(config, tree, job["block"], deadline, job["role"])
            elif stage == "preflight":
                model = Path(config["model_path"]).resolve(strict=True)
                actual = asset_digest(model)
                if actual != config["model_digest"]:
                    raise ValueError(f"Qwen asset digest drifted: {actual}")
                model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
                if model_config.get("model_type") != "qwen3" or model_config.get("vocab_size") != config["vocab_size"]:
                    raise ValueError("The supplied model assets are incompatible with this Qwen3 protocol")
                result = quality(config, tree, job["block"], deadline, "preflight")
                with Container(config, tree, deadline) as worker:
                    first = measure(worker, job["block"])
                    second = measure(worker, job["block"])
                    profiling = profile(worker, config)
                if not profiling["evidence"]["kineto_gpu_activity_count"]:
                    raise ValueError("No GPU activity observed during the SM121 profiling probe")
                result.update({"timing": [first, second], "model_digest": actual,
                               "image": config["image"], "engine_revision": repository.revision,
                               "profile": profiling["evidence"]})
            else:
                with Container(config, tree, deadline) as worker:
                    if stage == "measure":
                        result = measure(worker, job["block"])
                    elif stage == "profile":
                        result = profile(worker, config)
                    else:
                        raise ValueError("Unrecognized Spark worker stage")
        return {**result, "sensor_before": before, "sensor_after": sensor()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hash-model", type=Path)
    args = parser.parse_args()
    if args.hash_model:
        print(asset_digest(args.hash_model))
        return
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        job = json.load(sys.stdin)
        if job["configuration"] != digest(config):
            raise ValueError("Worker configuration differs from the coordinator lock")
        data = run(config, job)
        print(canonical({"status": "ok", "data": data}))
    except TimeoutError as failure:
        print(canonical({"status": "timeout", "detail": str(failure), "data": {}}))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as failure:
        print(canonical({"status": "failed", "detail": str(failure), "data": {}}))


if __name__ == "__main__":
    main()
