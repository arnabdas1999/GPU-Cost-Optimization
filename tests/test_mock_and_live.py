"""Mock vLLM server against the real client, and the live exporter during a run."""

import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest

from rollout_cost.config import load_config
from rollout_cost.contract import LIVE_METRICS
from rollout_cost.mock_server import MockVLLM, load_all_tasks
from rollout_cost.model_client import ModelError, OpenAICompatClient, RequestContext, ScriptedReply, build_scripts
from rollout_cost.runner import execute_run
from rollout_cost.telemetry import extract_signals, parse_prometheus

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("prometheus_client")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def start_mock(**kwargs):
    tasks = load_all_tasks(ROOT)
    mock = MockVLLM(build_scripts(tasks, "reference"), {t.title: t.task_id for t in tasks}, **kwargs)
    server = await asyncio.start_server(mock.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return mock, server, f"http://127.0.0.1:{port}"


def client(base: str) -> OpenAICompatClient:
    return OpenAICompatClient(base + "/v1", "coding-model", 0.2, 0.95, 10, 5, max_retries=0, retry_backoff_s=0)


def ctx(i=0):
    return RequestContext("r", "e", "t", f"req-{i}", 0, 1)


def test_mock_streams_scripted_reply_with_usage_and_metrics():
    async def go():
        mock, server, base = await start_mock(base_latency_s=0.01, per_token_latency_s=0.0)
        tasks = load_all_tasks(ROOT)
        task = tasks[0]
        c = client(base)
        try:
            messages = [{"role": "system", "content": "s"}, {"role": "user", "content": f"Task: {task.title}\n..."}]
            resp = await c.complete(messages, 512, ctx(), [])
            async with httpx.AsyncClient() as http:
                signals = extract_signals(parse_prometheus((await http.get(base + "/metrics")).text))
                version = (await http.get(base + "/version")).json()
        finally:
            await c.aclose()
            server.close()
        return resp, signals, version, task

    resp, signals, version, task = asyncio.run(go())
    assert json.loads(resp.content)["action"] == "read_file"  # first step of the reference policy
    assert resp.prompt_tokens and resp.completion_tokens and resp.ttft_s is not None
    assert version["mock"] is True
    assert signals["requests_running"] == 0 and signals["prompt_tokens_total"] == resp.prompt_tokens
    assert signals["prefix_cache_queries_total"] > 0


def test_mock_context_overflow_maps_to_context_error():
    async def go():
        _, server, base = await start_mock(max_model_len=64)
        c = client(base)
        try:
            with pytest.raises(ModelError) as info:
                await c.complete([{"role": "user", "content": "x" * 400}], 32, ctx(), [])
            return info.value.kind
        finally:
            await c.aclose()
            server.close()

    assert asyncio.run(go()) == "context_overflow"


def test_cancelled_request_is_aborted_by_server():
    async def go():
        mock, server, base = await start_mock(base_latency_s=30.0)
        c = client(base)
        task = asyncio.create_task(c.complete([{"role": "user", "content": "hi"}], 16, ctx(), []))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if mock.running:
                break
        assert mock.running == 1
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if mock.running == 0:
                break
        await c.aclose()
        server.close()
        return mock.running, mock.finished["abort"]

    running, aborted = asyncio.run(go())
    assert running == 0 and aborted == 1


def test_mock_queues_beyond_max_num_seqs():
    async def go():
        mock, server, base = await start_mock(max_num_seqs=1, base_latency_s=0.3)
        c = client(base)
        calls = [asyncio.create_task(c.complete([{"role": "user", "content": f"q{i}"}], 8, ctx(i), [])) for i in range(3)]
        await asyncio.sleep(0.15)
        observed = (mock.running, mock.waiting)
        await asyncio.gather(*calls)
        await c.aclose()
        server.close()
        return observed

    assert asyncio.run(go()) == (1, 2)


def test_live_exporter_serves_contract_metrics_during_run(tmp_path):
    port = free_port()
    cfg = load_config(ROOT / "configs" / "synthetic.yaml", {
        "model": {"scripted_policy": "reference", "scripted_base_latency_s": 0.05},
        "workload": {"repetitions": 1}, "telemetry": {"live_metrics_port": port, "live_metrics_host": "127.0.0.1"}})
    scrapes: list[str] = []

    async def probe():
        async with httpx.AsyncClient(timeout=1) as http:
            while True:
                await asyncio.sleep(0.2)
                try:
                    scrapes.append((await http.get(f"http://127.0.0.1:{port}/metrics")).text)
                except httpx.HTTPError:
                    pass

    async def go():
        prober = asyncio.create_task(probe())
        _, summary = await execute_run(cfg, "dev", tmp_path, run_id="live")
        prober.cancel()
        return summary

    summary = asyncio.run(go())
    assert scrapes, "exporter was never reachable"
    exposed = {line.split("{")[0].split(" ")[0] for text in scrapes for line in text.splitlines() if line.startswith("rc_")}
    always = {n for n, spec in LIVE_METRICS.items() if not n.startswith("rc_gpu") and n not in
              ("rc_episodes_terminal_total", "rc_llm_requests_total")}
    assert always <= exposed
    assert exposed <= set(LIVE_METRICS)  # nothing outside the contract
    actives = [float(l.split(" ")[-1]) for t in scrapes for l in t.splitlines() if l.startswith("rc_episodes_active{")]
    assert max(actives) <= cfg.execution.max_active_episodes
    assert summary["outcomes"]["success"] == 6
    # The exporter is stopped with the run, so the port is free again.
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_unknown_price_omits_cost_series(tmp_path):
    port = free_port()
    cfg = load_config(ROOT / "configs" / "synthetic.yaml", {
        "model": {"scripted_policy": "noop", "scripted_base_latency_s": 0.3}, "workload": {"repetitions": 1},
        "cost": {"allocation_hourly_rate": None}, "telemetry": {"live_metrics_port": port, "live_metrics_host": "127.0.0.1"}})
    texts: list[str] = []

    async def go():
        async def probe():
            async with httpx.AsyncClient(timeout=1) as http:
                while True:
                    await asyncio.sleep(0.1)
                    try:
                        texts.append((await http.get(f"http://127.0.0.1:{port}/metrics")).text)
                    except httpx.HTTPError:
                        pass

        prober = asyncio.create_task(probe())
        await execute_run(cfg, "dev", tmp_path, run_id="noprice")
        prober.cancel()

    asyncio.run(go())
    assert texts and not any("rc_allocation_cost{" in t for t in texts)


def test_scripted_error_reply_from_mock_is_server_error():
    async def go():
        tasks = load_all_tasks(ROOT)
        scripts = {t.task_id: [ScriptedReply("", error="server")] for t in tasks}
        mock = MockVLLM(scripts, {t.title: t.task_id for t in tasks}, base_latency_s=0.01)
        server = await asyncio.start_server(mock.handle, "127.0.0.1", 0)
        base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        c = client(base)
        try:
            with pytest.raises(ModelError) as info:
                await c.complete([{"role": "user", "content": f"Task: {tasks[0].title}"}], 16, ctx(), [])
            return info.value.kind
        finally:
            await c.aclose()
            server.close()

    assert asyncio.run(go()) == "server"
