"""Server readiness must belong to a live child in managed restart mode."""

import asyncio
import os
from pathlib import Path
import signal
import socket
import sys
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from rollout_cost.config import load_config
from rollout_cost.server import ServerController

ROOT = Path(__file__).resolve().parents[1]


def controller(tmp_path, port=8000):
    cfg = load_config(ROOT / "configs" / "mock-vllm.yaml", {
        "model": {"base_url": f"http://127.0.0.1:{port}/v1"}, "server": {"args": {"port": port}}})
    return ServerController(cfg, tmp_path, Mock(), None)


@pytest.mark.parametrize("exits_during_health", [False, True])
def test_healthy_endpoint_cannot_hide_a_dead_child(tmp_path, exits_during_health):
    ctrl = controller(tmp_path)
    ctrl._proc = Mock(returncode=23)
    ctrl._proc.poll.side_effect = [None, 23] if exits_during_health else [23]
    ctrl._get = AsyncMock(return_value=httpx.Response(200))
    with pytest.raises(RuntimeError, match="exited with code 23"):
        asyncio.run(ctrl.wait_healthy(1))
    assert ctrl._get.await_count == int(exits_during_health)


@pytest.mark.parametrize("managed", [False, True])
def test_live_server_can_become_ready(tmp_path, managed):
    ctrl = controller(tmp_path)
    if managed:
        ctrl._proc = Mock()
        ctrl._proc.poll.return_value = None
    ctrl._get = AsyncMock(return_value=httpx.Response(200))
    assert asyncio.run(ctrl.wait_healthy(1)) >= 0


def test_restart_refuses_an_occupied_port_without_launching_or_stopping_its_owner(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr("rollout_cost.server.subprocess.Popen", launch)

    async def go():
        def connected(reader, writer):
            writer.close()

        server = await asyncio.start_server(connected, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        ctrl = controller(tmp_path, port)
        try:
            with pytest.raises(RuntimeError, match="already listening after stop"):
                await ctrl.restart()
            # The unrelated listener is still reachable after rejection.
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(go())
    launch.assert_not_called()
    assert not (tmp_path / "server.log").exists()


@pytest.mark.parametrize("idle", [{"verified": False, "reason": "request gauges unavailable"}, {}])
def test_preparation_stops_before_the_workload_if_idleness_is_unverified(tmp_path, idle):
    ctrl = controller(tmp_path)
    ctrl.restart = AsyncMock(return_value={"method": "restart"})
    ctrl.warmup = AsyncMock(return_value={"requests": 0})
    ctrl.wait_idle = AsyncMock(return_value=idle)
    ctrl.snapshot = AsyncMock()
    with pytest.raises(RuntimeError, match="refusing to start the measured workload"):
        asyncio.run(ctrl.prepare())
    ctrl.snapshot.assert_not_awaited()
    ctrl.recorder.emit.assert_any_call("server_idle_before_run", **idle)


@pytest.mark.skipif(sys.platform != "linux", reason="Worker liveness check requires Linux /proc")
@pytest.mark.parametrize("parent_already_exited", [False, True])
def test_shutdown_kills_workers_even_if_the_parent_exits_first(tmp_path, parent_already_exited):
    worker_ready = tmp_path / "worker-pid"
    worker_code = (
        "import os, signal, sys, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {worker_code!r}, sys.argv[1]])\n"
        "while not Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
        + ("" if parent_already_exited else "time.sleep(30)\n"), encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctrl = controller(tmp_path, port)
    ctrl.cfg.server.start_command = [sys.executable, str(parent), str(worker_ready)]
    ctrl.wait_healthy = AsyncMock(return_value=0.0)

    def worker_running(pid):
        try:
            # A killed orphan can remain a zombie until the container reaps it.
            return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    async def go():
        worker_pid = None
        try:
            await ctrl.restart()
            assert os.getpgid(ctrl._proc.pid) == ctrl._proc.pid
            for _ in range(500):
                if worker_ready.exists():
                    text = worker_ready.read_text().strip()
                    if text:
                        worker_pid = int(text)
                        break
                await asyncio.sleep(0.01)
            assert worker_pid is not None, "worker failed to start"
            assert os.getpgid(worker_pid) == ctrl._proc.pid
            if parent_already_exited:
                ctrl._proc.wait(timeout=5)
            ctrl.shutdown()
            for _ in range(100):
                if not worker_running(worker_pid):
                    break
                await asyncio.sleep(0.01)
            assert not worker_running(worker_pid), "worker survived server shutdown"
        finally:
            ctrl.shutdown()
            if worker_pid is not None and worker_running(worker_pid):
                os.kill(worker_pid, signal.SIGKILL)

    asyncio.run(go())
