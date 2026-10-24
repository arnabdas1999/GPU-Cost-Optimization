"""OpenAI-compatible streaming client against a mocked vLLM endpoint."""

import asyncio
import json

import httpx
import pytest

from rollout_cost.model_client import ModelError, OpenAICompatClient, RequestContext

CTX = RequestContext("run", "ep", "task", "ep-s0", 0, 7)


def sse(*chunks, done=True) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def client(handler) -> OpenAICompatClient:
    return OpenAICompatClient("http://test/v1", "coding-model", 0.2, 0.95, 5, 5, max_retries=1, retry_backoff_s=0,
                              transport=httpx.MockTransport(handler))


def call(c, log=None):
    async def go():
        try:
            return await c.complete([{"role": "user", "content": "hi"}], 32, CTX, log if log is not None else [])
        finally:
            await c.aclose()

    return asyncio.run(go())


def test_streams_content_usage_and_request_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["request_id"] = request.headers["X-Request-Id"]
        body = sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},  # empty first event: not a content token
            {"choices": [{"delta": {"content": '{"action": '}}]},
            {"choices": [{"delta": {"content": '"finish"}'}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 6}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    resp = call(client(handler))
    assert resp.content == '{"action": "finish"}'
    assert (resp.finish_reason, resp.prompt_tokens, resp.completion_tokens) == ("stop", 11, 6)
    assert resp.ttft_s is not None and resp.ttft_s <= resp.duration_s
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert seen["seed"] == 7 and seen["max_tokens"] == 32 and seen["request_id"] == "ep-s0"


def test_missing_usage_stays_unknown():
    def handler(request):
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}))

    resp = call(client(handler))
    assert resp.prompt_tokens is None and resp.completion_tokens is None
    assert resp.finish_reason == "length"


def test_retries_declared_server_errors_and_records_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))

    log = []
    resp = call(client(handler), log)
    assert resp.content == "ok"
    assert [a["status"] for a in log] == ["server", "ok"]


def test_retry_budget_is_bounded():
    def handler(request):
        return httpx.Response(503, text="overloaded")

    log = []
    with pytest.raises(ModelError) as info:
        call(client(handler), log)
    assert info.value.kind == "server"
    assert len(log) == 2  # one attempt plus max_retries=1


def test_context_overflow_and_rejections_are_not_retried():
    def overflow(request):
        return httpx.Response(400, text="This model's maximum context length is 8192 tokens")

    log = []
    with pytest.raises(ModelError) as info:
        call(client(overflow), log)
    assert info.value.kind == "context_overflow" and len(log) == 1

    with pytest.raises(ModelError) as info:
        call(client(lambda r: httpx.Response(404, text="no such model")))
    assert info.value.kind == "rejected"
