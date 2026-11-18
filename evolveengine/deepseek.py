"""Bounded DeepSeek usage for Pi. The local proxy owns provider credentials."""

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .schema import Cost, canonical


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_):
        return None


@dataclass(frozen=True)
class Provider:
    base_url: str
    model: str
    key_env: str
    model_name: str
    max_turn_output: int
    fingerprint: str | None = None

    def __post_init__(self):
        url = urlsplit(self.base_url)
        local_test = url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost"}
        if ((url.scheme != "https" and not local_test) or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("Use a trusted HTTPS model endpoint")
        if not self.model or not self.model_name or type(self.max_turn_output) is not int or self.max_turn_output <= 0:
            raise ValueError("Versioned provider configuration is required")

    def request(self, url, *, payload=None, timeout=15):
        key = os.environ.get(self.key_env)
        if not key:
            raise ValueError(f"Missing provider credential environment variable {self.key_env}")
        endpoint = self.base_url.rstrip("/") + url
        return build_opener(NoRedirect()).open(Request(endpoint, payload, {
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
        }), timeout=timeout)

    def preflight(self):
        with self.request("/models") as response:
            for row in json.load(response)["data"]:
                if row["id"] == self.model and row.get("name") == self.model_name:
                    return {"id": row["id"], "name": row["name"], "owner": row.get("owned_by"),
                            "required_response_fingerprint": self.fingerprint}
        raise ValueError("Provider does not advertise the locked model ID and name")


class ProviderBudget:
    def __init__(self, config, limit, ready=None):
        self.config = config
        self.limit = limit
        self.ready = ready
        self.token = secrets.token_urlsafe(24)
        self.rows = []
        self.spent = Cost()
        self.deadline = time.monotonic() + limit.wall_ms / 1000
        self.failure = None
        self.settled = threading.Event()
        self.settled.set()
        self.request_lock = threading.Lock()

    def __enter__(self):
        state = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def handle(self):
                try:
                    super().handle()
                except ConnectionResetError:
                    self.close_connection = True

            def do_POST(self):
                with state.request_lock:
                    state.handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.03}, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    @property
    def local_url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def handle(self, handler):
        if (handler.path != "/chat/completions" or
                not secrets.compare_digest(handler.headers.get("Authorization", ""), "Bearer " + self.token)):
            handler.send_error(403)
            return
        if self.ready is not None and not self.ready.is_file():
            self.failure = "Pi did not initialize the tool guard"
            handler.send_error(403, self.failure)
            return
        timeout = self.deadline - time.monotonic()
        reserved = None
        self.settled.clear()
        try:
            length = int(handler.headers.get("Content-Length", "0"))
            if not 0 < length <= self.limit.tokens * 2 or timeout <= 0:
                raise ValueError("Provider request exceeds the operation lease")
            payload = json.loads(handler.rfile.read(length))
            if payload.get("model") != self.config.model or payload.get("stream") is not True:
                raise ValueError("Unexpected model or response mode")
            # Serialized byte length plus fixed framing is a conservative text-only bound.
            input_bound = len(canonical([payload["messages"], payload.get("tools", [])]).encode("utf-8")) + 128
            ceiling = min(self.config.max_turn_output, self.limit.tokens - self.spent.tokens - input_bound)
            if ceiling <= 0:
                raise ValueError("Provider input and output do not fit the remaining token reservation")
            payload.pop("max_completion_tokens", None)
            payload.update(max_tokens=ceiling, stream_options={"include_usage": True},
                           thinking={"type": "disabled"}, temperature=0.0)
            reserved = Cost(input_tokens=input_bound, output_tokens=ceiling)
            with self.config.request("/chat/completions", payload=canonical(payload).encode(), timeout=timeout) as response:
                body, usage, fingerprints, models = self.collect_stream(response, ceiling)
            if not usage:
                raise ValueError("Provider did not return streaming usage")
            cost = token_cost(usage)
            self.spent += cost
            reserved = None
            self.rows.append({"phase": "chat", "usage": usage, "models": sorted(models),
                              "fingerprints": sorted(fingerprints)})
            if models != {self.config.model}:
                raise ValueError("Provider response model drift")
            if self.config.fingerprint and fingerprints != {self.config.fingerprint}:
                raise ValueError("Provider response version fingerprint drift")
            if self.spent.tokens > self.limit.tokens or cost.output_tokens > ceiling or cost.input_tokens > input_bound:
                raise ValueError("Provider exceeded the enforced token reservation")
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.stream_started = True
            try:
                handler.wfile.write(body)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except HTTPError as failure:
            self.failure = f"Provider returned HTTP {failure.code}"
            failure.close()
            handler.send_error(502, self.failure)
        except (ValueError, KeyError, OSError) as failure:
            self.failure = str(failure)
            if not getattr(handler, "stream_started", False):
                handler.send_error(400, "Provider request failed")
        finally:
            if reserved is not None:
                self.spent += reserved
                self.rows.append({"phase": "chat", "status": "usage_uncertain", "charged": reserved.record()})
            self.settled.set()

    def collect_stream(self, response, maximum_tokens):
        # Pi is noninteractive. Buffer one turn so the metered reply can be
        # validated before any tool calls from it reach the agent.
        usage, fingerprints, models = {}, set(), set()
        body = bytearray()
        for line in response:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Provider response exceeded its deadline")
            if len(body) + len(line) > 2**20 + 256 * maximum_tokens:
                raise ValueError("Provider event stream exceeded the response-size reservation")
            if line.startswith(b"data: "):
                content = line[6:].strip()
                if content and content != b"[DONE]":
                    chunk = json.loads(content)
                    if chunk.get("usage"):
                        usage.update(chunk["usage"])
                    if chunk.get("system_fingerprint"):
                        fingerprints.add(chunk["system_fingerprint"])
                    if chunk.get("model"):
                        models.add(chunk["model"])
            body.extend(line)
        return body, usage, fingerprints, models


def token_cost(usage):
    return Cost(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
