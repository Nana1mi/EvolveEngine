"""Trusted pipe service inside the pinned, offline GPU image."""

import base64
import contextlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import zlib


WIRE = "EVOLVEENGINE "
MODEL_PATH = "/model"


def packed(tensor):
    raw = tensor.detach().to(dtype=__import__("torch").float32, device="cpu").contiguous().numpy()
    return base64.b64encode(zlib.compress(raw.astype("<f4", copy=False).tobytes())).decode("ascii")


def runtime_info():
    import torch

    if platform.machine() != "aarch64" or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one aarch64 DGX Spark GPU")
    if torch.cuda.get_device_capability(0) != (12, 1) or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected NVIDIA GB10 / SM121")
    return {"architecture": platform.machine(), "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "torch": str(torch.__version__),
            "cuda": torch.version.cuda}


class MiniRuntime:
    def __init__(self, options):
        import torch
        from minisgl.llm import LLM
        from minisgl.core import SamplingParams
        from importlib.metadata import version

        if not Path(inspect.getfile(LLM)).resolve().is_relative_to("/engine/python/minisgl"):
            raise RuntimeError("Mini-SGLang was not imported from the mounted engine snapshot")
        if platform.machine() != "aarch64":
            raise RuntimeError("Expected an aarch64 DGX Spark worker")
        if options.get("attention_backend") in (None, "auto"):
            raise ValueError("A verified attention_backend must be locked")
        if options.get("cache_type") != "naive" or options.get("max_running_req") != 1:
            raise ValueError("This workload uses uncached, single-request generation")
        self.torch, self.params = torch, SamplingParams

        class ObservedLLM(LLM):
            def offline_send_result(self, reply):
                super().offline_send_result(reply)
                for message in reply:
                    if not self.ticks or not self.ticks[-1][2]:
                        self.ticks.append((message.next_token, time.perf_counter(), message.finished))

        options = dict(options)
        # Mini-SGLang's local single-rank gloo uses a private container loopback.
        self.llm = ObservedLLM(model_path=MODEL_PATH, dtype=torch.bfloat16, **options)
        # Inspect CUDA only after mini-sglang initializes its own device and streams.
        self.info = {**runtime_info(), "triton": version("triton"), "flashinfer": version("flashinfer-python")}

    def token(self, request):
        prompt = request["prompt"]
        limit = request["max_new_tokens"]
        forced = request.get("forced")
        positions = set(request.get("positions", []))
        logits_rows = {}
        counter = 0
        sampler = self.llm.engine.sampler
        original = sampler.sample

        def observe(logits, args):
            nonlocal counter
            tokens = original(logits, args)
            if counter in positions:
                logits_rows[str(counter)] = packed(logits[0])
            if forced is not None and counter < len(forced):
                tokens = self.torch.full_like(tokens, forced[counter])
            counter += 1
            return tokens

        self.llm.ticks = []
        if forced is not None or positions:
            sampler.sample = observe
        beginning = time.perf_counter()
        try:
            self.llm.generate([prompt], self.params(temperature=0, max_tokens=limit,
                                                     ignore_eos=request["ignore_eos"]))
        finally:
            if forced is not None or positions:
                sampler.sample = original
        tokens = self.llm.ticks
        if not tokens or not tokens[-1][2]:
            raise RuntimeError("Generation did not return a complete request")
        ids = [item[0] for item in tokens]
        finish = "eos" if not request["ignore_eos"] and ids[-1] == self.llm.eos_token_id else "length"
        return {"input_tokens": len(prompt), "output_tokens": len(ids), "ids": ids,
                "finish_reason": finish, "logits": logits_rows,
                "elapsed_seconds": time.perf_counter() - beginning,
                "ttft_seconds": tokens[0][1] - beginning,
                "tpot_seconds": (tokens[-1][1] - tokens[0][1]) / max(1, len(tokens) - 1)}

    def profile(self, request):
        from torch.profiler import ProfilerActivity, profile

        self.token(request)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            measured = self.token(request)
        with tempfile.TemporaryDirectory() as folder:
            filename = os.path.join(folder, "trace.json")
            trace.export_chrome_trace(filename)
            with open(filename, encoding="utf-8") as stream:
                events = json.load(stream)["traceEvents"]
        gpu_events = [event for event in events
                      if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
                      and event.get("ph") == "X" and event.get("dur", 0) > 0]
        return {"trace": gpu_events, "measurement": measured}

    def close(self):
        self.llm.shutdown()


class ReferenceRuntime:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.info = runtime_info()
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="eager",
        ).eval().to("cuda")
        self.eos = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True).eos_token_id

    def token(self, request):
        torch = self.torch
        input_ids = torch.tensor([request["prompt"]], device="cuda")
        past, tokens, rows = None, [], {}
        picks = set(request.get("positions", []))
        finish = "length"
        with torch.inference_mode():
            for step in range(request["max_new_tokens"]):
                answer = self.model(input_ids, past_key_values=past, use_cache=True)
                logits = answer.logits[0, -1]
                if step in picks:
                    rows[str(step)] = packed(logits)
                token = int(logits.argmax().item())
                tokens.append(token)
                if not request["ignore_eos"] and token == self.eos:
                    finish = "eos"
                    break
                past = answer.past_key_values
                input_ids = torch.tensor([[token]], device="cuda")
        return {"ids": tokens, "logits": rows, "finish_reason": finish}

    def close(self):
        pass


def serve():
    wire = sys.stdout

    def send(value):
        wire.write(WIRE + json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        wire.flush()

    init = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        runtime = ReferenceRuntime() if init["reference"] else MiniRuntime(init["engine_options"])
        send({"ready": runtime.info})
        try:
            for line in sys.stdin:
                request = json.loads(line)
                if request.get("shutdown"):
                    break
                reply = runtime.profile(request["input"]) if request.get("profile") else runtime.token(request["input"])
                send(reply)
        finally:
            runtime.close()


if __name__ == "__main__":
    serve()
