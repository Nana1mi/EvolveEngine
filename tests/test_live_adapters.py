import base64
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

from evolveengine.deepseek import Provider, ProviderBudget
from evolveengine.pi import PiAgent
from evolveengine.schema import Cost
from evolveengine.spark_worker import logit_checks


def provider_reply(request):
    tools = request.get("tools", [])
    turns = sum(item["role"] == "tool" for item in request["messages"])
    finish = "stop"
    delta = {"role": "assistant", "content": '{"ready":true}'}
    if tools and turns < 3:
        path = "outside.txt" if turns == 0 else "python/minisgl/runtime.py"
        args = {"path": path}
        tool = "read"
        if turns == 2:
            args["edits"] = [{"oldText": "speed = 1", "newText": "speed = 2"}]
            tool = "edit"
        delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{turns}",
                 "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}
        finish = "tool_calls"
    root = {"id": "test", "object": "chat.completion.chunk", "created": 1,
            "model": request["model"], "system_fingerprint": "fixture-v1"}
    chunks = [
        {**root, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**root, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {**root, "choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}},
    ]
    body = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    return body + b"data: [DONE]\n\n"


class LiveAdapterTests(unittest.TestCase):
    def setUp(self):
        self.provider = Provider("http://127.0.0.1:1", "deepseek-flash",
                                 "EVOLVE_PROVIDER_TEST_KEY", "DeepSeek-V4.1-Flash", 128, "fixture-v1")

    def test_provider_caps_carried_into_usage(self):
        cap = Cost(wall_ms=3000, input_tokens=240)
        upstream_requests = []

        def upstream(_url, *, payload, timeout):
            request = json.loads(payload)
            upstream_requests.append(request)
            return io.BytesIO(provider_reply(request))

        with patch.object(Provider, "request", side_effect=upstream), ProviderBudget(self.provider, cap) as broker:
            payload = {"model": "deepseek-flash", "stream": True, "max_tokens": 10000,
                       "messages": [{"role": "user", "content": "test"}]}

            def send(value):
                class Handler:
                    path = "/chat/completions"
                    send_header = end_headers = lambda *unused: None

                    def send_response(self, status):
                        self.status = status

                    def send_error(self, status, _detail=None):
                        self.status = status

                request = Handler()
                body = json.dumps(value).encode()
                request.headers = {"Authorization": "Bearer " + broker.token, "Content-Length": str(len(body))}
                request.rfile, request.wfile = io.BytesIO(body), io.BytesIO()
                broker.handle(request)
                return request.status, request.wfile.getvalue()

            status, stream = send(payload)
            self.assertEqual(status, 200)
            self.assertIn(b"[DONE]", stream)
            self.assertEqual(broker.spent.tokens, 25)
            self.assertLess(upstream_requests[0]["max_tokens"], 128)
            status, _ = send({**payload, "messages": [{"role": "user", "content": "X" * 230}]})
            self.assertEqual(status, 400)
            self.assertEqual(len(upstream_requests), 1)

    @unittest.skipUnless("EVOLVE_PI_TEST_COMMAND" in os.environ, "optional pinned Pi CLI")
    def test_actual_pi_guard_and_token_meter(self):
        pi = PiAgent(json.loads(os.environ["EVOLVE_PI_TEST_COMMAND"]), "1.5.2", self.provider)
        def upstream(_url, *, payload, timeout):
            return io.BytesIO(provider_reply(json.loads(payload)))

        with (TemporaryDirectory() as root,
              patch.object(Provider, "request", side_effect=upstream),
              ProviderBudget(self.provider, Cost(25000, input_tokens=16000)) as broker):
            tree = Path(root)
            source = tree / "python/minisgl/runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text("speed = 1\n")
            (tree / "outside.txt").write_text("PRIVATE_SENTINEL")
            scope = ("python/minisgl",)
            result, trace = pi.session(tree, "Apply the edit", broker, editing=scope, read_scope=scope)
            self.assertEqual(result, {"ready": True})
            self.assertEqual(source.read_text(), "speed = 2\n")
            results = [(item["toolName"], item["isError"]) for item in trace
                       if item["type"] == "tool_execution_end"]
            self.assertEqual(results, [("read", True), ("read", False), ("edit", False)])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(trace))
            self.assertEqual(broker.spent.tokens, 4 * 25)


class NumericalChecksTests(unittest.TestCase):
    def test_full_vocab_kl_and_margin(self):
        import struct
        try:
            __import__("numpy")
        except ImportError:
            self.skipTest("optional numerical worker dependency")

        def rows(*logits):
            binary = struct.pack("<" + "f" * len(logits), *logits)
            return {"0": base64.b64encode(zlib.compress(binary))}

        reference = rows(1.0, 0.0, -1.0)
        self.assertEqual(logit_checks(reference, reference, 3), (True, True, 0.0))
        self.assertFalse(logit_checks(reference, rows(1.0, 0.0), 3)[0])
        full, top, divergence = logit_checks(reference, rows(0.0, 1.0, -1.0), 3)
        self.assertTrue(full)
        self.assertFalse(top)
        self.assertGreater(divergence, 0.01)
