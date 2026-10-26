"""Model clients: a streaming OpenAI-compatible client and a scripted fake.

Cancellation: the harness bounds every call with a timeout. Cancelling a call
closes the HTTP stream, which makes vLLM abort the request. The runner still
verifies after the cohort that the server reports no running or waiting
requests before it closes a run.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({502, 503, 504})
CONTEXT_OVERFLOW_MARKERS = ("maximum context length", "context length", "too long", "max_model_len")


@dataclass(frozen=True)
class RequestContext:
    run_id: str
    episode_id: str
    task_id: str
    request_id: str
    step: int
    seed: int | None
    response_schema: dict | None = None


@dataclass
class ModelResponse:
    content: str
    finish_reason: str | None
    prompt_tokens: int | None  # None when the server did not report usage
    completion_tokens: int | None
    ttft_s: float | None  # first non-empty content token, from request start
    duration_s: float


class ModelError(Exception):
    """kind: transport | server | rejected | context_overflow"""

    def __init__(self, kind: str, message: str, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status


class OpenAICompatClient:
    def __init__(self, base_url: str, served_name: str, temperature: float, top_p: float,
                 request_timeout_s: float, connect_timeout_s: float, max_retries: int, retry_backoff_s: float,
                 api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.served_name = served_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        timeout = httpx.Timeout(connect=connect_timeout_s, read=request_timeout_s, write=30.0, pool=None)
        self._http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport,
                                       limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))

    def describe(self) -> dict:
        return {"kind": "openai", "base_url": self.base_url, "served_name": self.served_name,
                "temperature": self.temperature, "top_p": self.top_p, "max_retries": self.max_retries,
                "retry_on": ["connect_error", *sorted(RETRYABLE_STATUS)], "streaming": True}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, messages: list[dict], max_tokens: int, ctx: RequestContext, attempt_log: list[dict]) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            record: dict[str, Any] = {"attempt": attempt, "started_utc": _utc(), "status": "started"}
            attempt_log.append(record)
            started = time.monotonic()
            try:
                response = await self._stream_once(messages, max_tokens, ctx, record)
            except ModelError as exc:
                record.update(status=exc.kind, error=str(exc)[:500], http_status=exc.status, duration_s=time.monotonic() - started)
                if exc.retryable and attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_s)
                    continue
                raise
            except asyncio.CancelledError:
                record.update(status="cancelled", duration_s=time.monotonic() - started)
                raise
            record.update(status="ok", duration_s=response.duration_s, finish_reason=response.finish_reason)
            return response
        raise AssertionError("unreachable")

    async def _stream_once(self, messages: list[dict], max_tokens: int, ctx: RequestContext, record: dict) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.served_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if ctx.seed is not None:
            payload["seed"] = ctx.seed
        if ctx.response_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "rollout-action", "schema": ctx.response_schema}}
        started = time.monotonic()
        parts: list[str] = []
        finish_reason = None
        usage = None
        ttft = None
        try:
            async with self._http.stream("POST", "/chat/completions", json=payload,
                                         headers={"X-Request-Id": ctx.request_id}) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:2000]
                    if resp.status_code == 400 and any(m in body.lower() for m in CONTEXT_OVERFLOW_MARKERS):
                        raise ModelError("context_overflow", body, status=400)
                    if resp.status_code >= 500 or resp.status_code == 429:
                        raise ModelError("server", body, retryable=resp.status_code in RETRYABLE_STATUS, status=resp.status_code)
                    raise ModelError("rejected", body, status=resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise ModelError("server", json.dumps(chunk["error"])[:500])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            if ttft is None:
                                ttft = time.monotonic() - started
                            parts.append(text)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except httpx.ConnectError as exc:
            raise ModelError("transport", f"connect error: {exc}", retryable=True) from exc
        except (httpx.TransportError, json.JSONDecodeError) as exc:
            # Mid-stream failures are not retried: the attempt may have produced work.
            raise ModelError("transport", f"{type(exc).__name__}: {exc}", retryable=False) from exc
        record["ttft_s"] = ttft
        return ModelResponse(
            content="".join(parts),
            finish_reason=finish_reason,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            ttft_s=ttft,
            duration_s=time.monotonic() - started,
        )


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Scripted fake (synthetic: verifies software behavior, not GPU efficiency)


@dataclass
class ScriptedReply:
    content: str
    finish_reason: str = "stop"
    error: str | None = None  # raise ModelError(kind=error)
    hang: bool = False  # never return (exercises timeouts and cancellation)


@dataclass
class ScriptedModelClient:
    scripts: dict[str, list[ScriptedReply]]
    base_latency_s: float = 0.2
    per_token_latency_s: float = 0.002
    # Synthetic slowdown per additional in-flight request, to mimic batching contention.
    contention: float = 0.3
    chars_per_token: float = 3.5
    inflight: int = 0
    max_inflight: int = 0
    calls: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {"kind": "scripted", "synthetic": True, "base_latency_s": self.base_latency_s,
                "per_token_latency_s": self.per_token_latency_s, "contention": self.contention}

    async def aclose(self) -> None:
        return None

    async def complete(self, messages: list[dict], max_tokens: int, ctx: RequestContext, attempt_log: list[dict]) -> ModelResponse:
        step = sum(1 for m in messages if m["role"] == "assistant")
        script = self.scripts.get(ctx.task_id, [])
        reply = script[step] if step < len(script) else ScriptedReply('{"action": "finish"}')
        record: dict[str, Any] = {"attempt": 0, "started_utc": _utc(), "status": "started"}
        attempt_log.append(record)
        self.calls.append(ctx.request_id)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        started = time.monotonic()
        try:
            content, finish = reply.content, reply.finish_reason
            tokens = max(1, math.ceil(len(content) / self.chars_per_token))
            if tokens > max_tokens:
                content, finish, tokens = content[: int(max_tokens * self.chars_per_token)], "length", max_tokens
            first = self.base_latency_s * (1 + self.contention * (self.inflight - 1))
            await asyncio.sleep(first)
            if reply.hang:
                await asyncio.Event().wait()
            if reply.error:
                raise ModelError(reply.error, f"scripted {reply.error}", status=500 if reply.error == "server" else None)
            ttft = time.monotonic() - started
            await asyncio.sleep(tokens * self.per_token_latency_s * (1 + self.contention * (self.inflight - 1)))
        except ModelError as exc:
            record.update(status=exc.kind, duration_s=time.monotonic() - started)
            raise
        except asyncio.CancelledError:
            record.update(status="cancelled", duration_s=time.monotonic() - started)
            raise
        finally:
            self.inflight -= 1
        prompt_tokens = sum(math.ceil(len(m["content"]) / self.chars_per_token) + 4 for m in messages)
        duration = time.monotonic() - started
        record.update(status="ok", duration_s=duration, finish_reason=finish, ttft_s=ttft)
        return ModelResponse(content, finish, prompt_tokens, tokens, ttft, duration)


def build_scripts(tasks, policy: str) -> dict[str, list[ScriptedReply]]:
    """Deterministic scripted policies for exercising the harness.

    * ``reference``: read the source file, write the reference fix, run tests, finish.
    * ``noop``: finish immediately (every task should fail grading).
    * ``mixed``: by task position, cycle through a reference solve, a malformed
      reply followed by a solve, and an attempt that never fixes anything.
    """
    scripts: dict[str, list[ScriptedReply]] = {}
    for position, task in enumerate(tasks):
        source = next(f for f in task.initial_files if any(f.startswith(p.rstrip("/") + "/") or f == p for p in task.graded_paths))
        fixed = (Path(task.grader_dir) / "reference" / source).read_text(encoding="utf-8")
        read = ScriptedReply(json.dumps({"action": "read_file", "path": source}))
        write = ScriptedReply(json.dumps({"action": "write_file", "path": source, "content": fixed}))
        test = ScriptedReply(json.dumps({"action": "run_tests"}))
        finish = ScriptedReply(json.dumps({"action": "finish"}))
        solve = [read, write, test, finish]
        if policy == "reference":
            scripts[task.task_id] = solve
        elif policy == "noop":
            scripts[task.task_id] = [finish]
        elif policy == "mixed":
            variant = position % 3
            if variant == 0:
                scripts[task.task_id] = solve
            elif variant == 1:
                scripts[task.task_id] = [ScriptedReply("I will read the file first."), *solve]
            else:
                scripts[task.task_id] = [read, test, test, finish]
        else:
            raise ValueError(f"unknown scripted policy {policy!r}")
    return scripts


def make_client(cfg, tasks):
    m = cfg.model
    if m.kind == "scripted":
        return ScriptedModelClient(
            build_scripts(tasks, m.scripted_policy),
            base_latency_s=m.scripted_base_latency_s,
            per_token_latency_s=m.scripted_per_token_latency_s,
            contention=m.scripted_contention,
        )
    return OpenAICompatClient(
        base_url=m.base_url,
        served_name=m.served_name,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        request_timeout_s=m.request_timeout_s,
        connect_timeout_s=m.connect_timeout_s,
        max_retries=m.max_request_retries,
        retry_backoff_s=m.retry_backoff_s,
        api_key=os.environ.get(m.api_key_env) if m.api_key_env else None,
    )
