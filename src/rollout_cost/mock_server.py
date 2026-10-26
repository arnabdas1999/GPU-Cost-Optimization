"""A mock vLLM server: the OpenAI-compatible endpoints and metrics the runner uses.

SYNTHETIC. It lets the real client path (streaming HTTP, usage, retries,
context overflow), the server lifecycle (restart, health, warm-up, cache reset,
drain) and the vLLM dashboard panels run without a GPU. Replies come from the
same scripted policies as the in-process fake; latency, batching contention,
``max-num-seqs`` queueing and prefix-cache reuse are crude emulations.

``/version`` reports ``"mock": true`` and the runner labels such runs synthetic.

Run: ``python -m rollout_cost.mock_server --port 8000 --policy mixed``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .fixtures import load_task
from .model_client import ScriptedReply, build_scripts

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
FINISH = ScriptedReply('{"action": "finish"}')


class _Disconnected(Exception):
    pass


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def lines(self, name: str, labels: str) -> list[str]:
        out = [f"# TYPE {name} histogram"]
        for bound, count in zip(self.buckets, self.counts):
            out.append(f'{name}_bucket{{{labels},le="{bound}"}} {count}')
        out += [f'{name}_bucket{{{labels},le="+Inf"}} {self.total}', f"{name}_sum{{{labels}}} {self.sum}",
                f"{name}_count{{{labels}}} {self.total}"]
        return out


class MockVLLM:
    def __init__(self, scripts: dict[str, list[ScriptedReply]], titles: dict[str, str], served_name: str = "coding-model",
                 max_model_len: int = 8192, max_num_seqs: int = 8, base_latency_s: float = 0.15,
                 per_token_latency_s: float = 0.002, contention: float = 0.3, enable_prefix_caching: bool = True,
                 chars_per_token: float = 3.5):
        self.scripts = scripts
        self.titles = titles
        self.served_name = served_name
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.base_latency_s = base_latency_s
        self.per_token_latency_s = per_token_latency_s
        self.contention = contention
        self.enable_prefix_caching = enable_prefix_caching
        self.chars_per_token = chars_per_token
        self._slots = asyncio.Semaphore(max_num_seqs)
        self.running = 0
        self.waiting = 0
        self.counters: dict[str, float] = defaultdict(float)
        self.finished: dict[str, int] = defaultdict(int)
        self.ttft = _Histogram(TTFT_BUCKETS)
        self.e2e = _Histogram(E2E_BUCKETS)
        self._prefixes: set[str] = set()

    # -- token and cache emulation ------------------------------------------

    def tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def prompt_tokens(self, messages: list[dict]) -> int:
        return sum(self.tokens(m.get("content", "")) + 4 for m in messages)

    @staticmethod
    def _prefix_keys(messages: list[dict]) -> list[str]:
        digest = hashlib.sha256()
        keys = []
        for m in messages:
            digest.update(json.dumps([m.get("role"), m.get("content")]).encode())
            keys.append(digest.copy().hexdigest())
        return keys

    def _account_prefix(self, messages: list[dict]) -> None:
        keys = self._prefix_keys(messages)
        hit = 0
        if self.enable_prefix_caching:
            for i, key in enumerate(keys):
                if key in self._prefixes:
                    hit = self.prompt_tokens(messages[: i + 1])
            self._prefixes.update(keys)
        self.counters["prefix_queries"] += self.prompt_tokens(messages)
        self.counters["prefix_hits"] += hit

    def _remember_reply(self, messages: list[dict], reply: str) -> None:
        if self.enable_prefix_caching:
            self._prefixes.update(self._prefix_keys([*messages, {"role": "assistant", "content": reply}]))

    def reset_prefix_cache(self) -> None:
        self._prefixes.clear()

    def choose_reply(self, messages: list[dict]) -> ScriptedReply:
        first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        title = first_user.split("\n", 1)[0].removeprefix("Task: ").strip()
        task_id = self.titles.get(title)
        if task_id is None:
            return FINISH  # warm-up, doctor and other non-task prompts
        step = sum(1 for m in messages if m.get("role") == "assistant")
        script = self.scripts.get(task_id, [])
        return script[step] if step < len(script) else FINISH

    # -- metrics -------------------------------------------------------------

    def metrics_text(self) -> str:
        labels = f'model_name="{self.served_name}"'
        c = self.counters
        lines = []
        for name, kind, value in (
            ("vllm:num_requests_running", "gauge", self.running),
            ("vllm:num_requests_waiting", "gauge", self.waiting),
            ("vllm:kv_cache_usage_perc", "gauge", min(1.0, self.running / self.max_num_seqs * 0.6)),
            ("vllm:prefix_cache_queries_total", "counter", c["prefix_queries"]),
            ("vllm:prefix_cache_hits_total", "counter", c["prefix_hits"]),
            ("vllm:prompt_tokens_total", "counter", c["prompt_tokens"]),
            ("vllm:generation_tokens_total", "counter", c["generation_tokens"]),
            ("vllm:num_preemptions_total", "counter", 0),
        ):
            lines += [f"# TYPE {name} {kind}", f"{name}{{{labels}}} {value}"]
        lines.append("# TYPE vllm:request_success_total counter")
        for reason, count in sorted(self.finished.items()):
            lines.append(f'vllm:request_success_total{{{labels},finished_reason="{reason}"}} {count}')
        lines += self.ttft.lines("vllm:time_to_first_token_seconds", labels)
        lines += self.e2e.lines("vllm:e2e_request_latency_seconds", labels)
        return "\n".join(lines) + "\n"

    # -- HTTP ----------------------------------------------------------------

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            method, target, _ = request_line.decode("latin-1").split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = line.decode("latin-1").partition(":")
                headers[key.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0") or 0))
            await self.route(method, target.split("?", 1)[0], body, reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError, _Disconnected, ValueError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _respond(self, writer, status: int, payload, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else (
            payload.encode() if isinstance(payload, str) else json.dumps(payload).encode())
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
        head = (f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n")
        writer.write(head.encode() + data)
        await writer.drain()

    async def route(self, method: str, path: str, body: bytes, reader, writer) -> None:
        if method == "GET" and path == "/health":
            await self._respond(writer, 200, b"", "text/plain")
        elif method == "GET" and path == "/version":
            await self._respond(writer, 200, {"version": "mock-1", "mock": True})
        elif method == "GET" and path == "/v1/models":
            await self._respond(writer, 200, {"object": "list", "data": [{"id": self.served_name, "object": "model",
                                                                            "max_model_len": self.max_model_len}]})
        elif method == "GET" and path == "/metrics":
            await self._respond(writer, 200, self.metrics_text(), "text/plain; version=0.0.4")
        elif method == "POST" and path == "/reset_prefix_cache":
            self.reset_prefix_cache()
            await self._respond(writer, 200, b"", "text/plain")
        elif method == "POST" and path == "/v1/chat/completions":
            await self.chat(json.loads(body or b"{}"), reader, writer)
        else:
            await self._respond(writer, 404, {"error": {"message": f"no route {method} {path}"}})

    async def chat(self, req: dict, reader, writer) -> None:
        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or 256)
        prompt_tokens = self.prompt_tokens(messages)
        if req.get("model") not in (None, self.served_name):
            await self._respond(writer, 404, {"error": {"message": f"The model `{req.get('model')}` does not exist."}})
            return
        if prompt_tokens + max_tokens > self.max_model_len:
            await self._respond(writer, 400, {"error": {"message": (
                f"This model's maximum context length is {self.max_model_len} tokens. However, you requested "
                f"{prompt_tokens + max_tokens} tokens ({prompt_tokens} in the messages, {max_tokens} in the completion).")}})
            return
        reply = self.choose_reply(messages)
        content, finish = reply.content, reply.finish_reason
        completion_tokens = self.tokens(content)
        if completion_tokens > max_tokens:
            content, finish, completion_tokens = content[: int(max_tokens * self.chars_per_token)], "length", max_tokens

        eof = asyncio.ensure_future(reader.read(1))  # completes when the client disconnects
        started = time.monotonic()
        self.waiting += 1
        try:
            await self._wait_or_disconnect(self._slots.acquire(), eof)
        except _Disconnected:
            self.finished["abort"] += 1
            raise
        finally:
            self.waiting -= 1
        self.running += 1
        self._account_prefix(messages)
        self.counters["prompt_tokens"] += prompt_tokens
        try:
            await self._sleep(self.base_latency_s * self._slowdown(), eof)
            if reply.error:
                await self._respond(writer, 500, {"error": {"message": "mock internal error"}})
                return
            self.ttft.observe(time.monotonic() - started)
            if not req.get("stream"):
                await self._sleep(completion_tokens * self.per_token_latency_s * self._slowdown(), eof)
                await self._respond(writer, 200, {
                    "id": "mock", "object": "chat.completion", "model": self.served_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens}})
            else:
                await self._stream(writer, eof, content, finish, prompt_tokens, completion_tokens,
                                   bool((req.get("stream_options") or {}).get("include_usage")))
            self.counters["generation_tokens"] += completion_tokens
            self.finished[finish] += 1
            self._remember_reply(messages, content)
            self.e2e.observe(time.monotonic() - started)
        except (_Disconnected, ConnectionError):
            self.finished["abort"] += 1
            raise
        finally:
            self.running -= 1
            self._slots.release()
            eof.cancel()

    def _slowdown(self) -> float:
        return 1 + self.contention * max(0, self.running - 1)

    async def _wait_or_disconnect(self, awaitable, eof) -> None:
        task = asyncio.ensure_future(awaitable)
        done, _ = await asyncio.wait({task, eof}, return_when=asyncio.FIRST_COMPLETED)
        if task not in done:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:  # acquired just as the client left: give the slot back
                self._slots.release()
            raise _Disconnected()

    async def _sleep(self, seconds: float, eof) -> None:
        await self._wait_or_disconnect(asyncio.sleep(seconds), eof)

    async def _stream(self, writer, eof, content: str, finish: str, prompt_tokens: int, completion_tokens: int,
                      include_usage: bool) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                     b"Connection: close\r\n\r\n")

        def event(obj) -> bytes:
            return f"data: {json.dumps(obj)}\n\n".encode()

        base = {"id": "mock", "object": "chat.completion.chunk", "model": self.served_name}
        writer.write(event({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}))
        await writer.drain()
        piece = max(1, int(4 * self.chars_per_token))  # ~4 tokens per chunk
        chunks = [content[i: i + piece] for i in range(0, len(content), piece)] or [""]
        for i, chunk in enumerate(chunks):
            if i:
                await self._sleep(4 * self.per_token_latency_s * self._slowdown(), eof)
            last = i == len(chunks) - 1
            writer.write(event({**base, "choices": [{"index": 0, "delta": {"content": chunk},
                                                     "finish_reason": finish if last else None}]}))
            await writer.drain()
        if include_usage:
            writer.write(event({**base, "choices": [], "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}}))
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()


def load_all_tasks(root: Path) -> list:
    tasks_dir, graders_dir = root / "fixtures" / "tasks", root / "graders"
    return [load_task(d, graders_dir) for d in sorted(tasks_dir.iterdir()) if (d / "task.yaml").exists()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rollout-cost mock-server", description="Mock vLLM server (SYNTHETIC).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default="coding-model")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--policy", default="mixed", choices=["reference", "noop", "mixed"])
    p.add_argument("--base-latency", type=float, default=0.15)
    p.add_argument("--per-token-latency", type=float, default=0.002)
    p.add_argument("--contention", type=float, default=0.3)
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--project-root", default=".")
    return p


async def serve(args) -> None:
    tasks = load_all_tasks(Path(args.project_root))
    mock = MockVLLM(build_scripts(tasks, args.policy), {t.title: t.task_id for t in tasks},
                    served_name=args.served_model_name, max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs, base_latency_s=args.base_latency,
                    per_token_latency_s=args.per_token_latency, contention=args.contention,
                    enable_prefix_caching=args.enable_prefix_caching)
    server = await asyncio.start_server(mock.handle, args.host, args.port)
    print(f"mock vLLM (SYNTHETIC) serving {args.served_model_name} on http://{args.host}:{args.port}/v1", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
