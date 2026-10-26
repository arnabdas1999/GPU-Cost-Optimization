"""Preflight rejects servers whose idleness cannot be measured."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from rollout_cost import cli
from rollout_cost.config import load_config
from rollout_cost.model_client import ModelResponse

ROOT = Path(__file__).resolve().parents[1]
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"


@pytest.mark.parametrize("metrics,expected_failures", [
    ("", 1),
    (f"{RUNNING} 0\n", 1),
    (f"{WAITING} 0\n", 1),
    (f"{RUNNING} NaN\n{WAITING} 0\n", 1),
    (f"{RUNNING} 0\n{WAITING} +Inf\n", 1),
    (f"{RUNNING} -1\n{WAITING} 0\n", 1),
    ("http-error", 1),
    ("unreachable", 1),
    (f"{RUNNING} 0\n{WAITING} 0\n", 0),
    (f"{RUNNING} 1\n{WAITING} 2\n", 0),
])
def test_doctor_requires_usable_idle_gauges(monkeypatch, metrics, expected_failures, capsys):
    cfg = load_config(ROOT / "configs/mock-vllm.yaml")

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": cfg.model.served_name}]})
        if metrics == "unreachable":
            raise httpx.ConnectError("metrics unavailable", request=request)
        return httpx.Response(404 if metrics == "http-error" else 200, text=metrics)

    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    model = Mock()
    model.complete = AsyncMock(return_value=ModelResponse('{"action":"finish"}', "stop", 2, 2, 0.01, 0.02))
    model.aclose = AsyncMock()
    monkeypatch.setattr(cli, "make_client", lambda *args: model)

    assert asyncio.run(cli._doctor_endpoint(cfg)) == expected_failures
    output = capsys.readouterr().out
    assert ("[FAIL]" in output) == bool(expected_failures)
    model.aclose.assert_awaited_once()


def test_doctor_cancellation_during_startup_always_stops_server(monkeypatch, tmp_path):
    cfg = load_config(ROOT / "configs/mock-vllm.yaml")
    controller = Mock()
    controller.restart = AsyncMock(side_effect=asyncio.CancelledError())
    controller._log = (tmp_path / "server.log").open("w")
    monkeypatch.setattr("rollout_cost.server.ServerController", lambda *args: controller)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cli._doctor_server(cfg, start_server=True))
    controller._stop.assert_called_once()
    assert controller._log.closed
