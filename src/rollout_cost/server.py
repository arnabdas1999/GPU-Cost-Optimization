"""Model-server lifecycle around a measured run.

Before every measured run the same procedure is applied, so baseline and
candidate start from comparable cache state regardless of trial order:

1. Reset: restart the server (``cache_reset: restart``) or call
   ``POST /reset_prefix_cache`` (``cache_reset: endpoint``).
2. Wait for ``/health``.
3. Warm up with fixed prompts that do not share the harness system prompt, so
   the measured task prefixes start uncached in every condition.
4. Wait until the server reports no running or waiting requests.

Prefix-cache counters are snapshotted after warm-up and after the cohort, so
the report can show how much reuse happened inside the measured window.
After the cohort, the run is not closed until the server is idle again;
otherwise a cancelled request could still be consuming the GPU during the next
trial. If idleness cannot be verified, the run is flagged.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import Config, render_command
from .model_client import RequestContext
from .protocol import whole_file_schema
from .telemetry import EventRecorder, extract_signals, fetch_metrics, parse_prometheus, server_root


class ServerController:
    def __init__(self, cfg: Config, run_dir: Path, recorder: EventRecorder, client):
        self.cfg = cfg
        self.run_dir = run_dir
        self.recorder = recorder
        self.client = client
        self.root = server_root(cfg.model.base_url)
        self._proc: subprocess.Popen | None = None
        self._log = None

    @property
    def command(self) -> list[str] | None:
        s = self.cfg.server
        if not s.start_command:
            return None
        # "python" means the runner's interpreter, as in the local tool executor.
        return [sys.executable if part == "python" else part for part in render_command(s.start_command, s.args)]

    async def _get(self, path: str, timeout: float = 5.0) -> httpx.Response | None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                return await http.get(self.root + path)
        except httpx.HTTPError:
            return None

    async def wait_healthy(self, timeout_s: float) -> float:
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            self._check_process()
            resp = await self._get("/health")
            # The child can exit while the HTTP request is in progress. A
            # response from another process must not hide that startup failure.
            self._check_process()
            if resp is not None and resp.status_code == 200:
                return time.monotonic() - started
            await asyncio.sleep(2.0)
        raise RuntimeError(f"server not healthy after {timeout_s:.0f}s")

    def _check_process(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            raise RuntimeError(f"server process exited with code {self._proc.returncode}; see server.log")

    async def _require_unused_endpoint(self) -> None:
        endpoint = httpx.URL(self.root)
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(endpoint.host, port), timeout=2.0)
        except (OSError, TimeoutError):
            return
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        raise RuntimeError(f"server endpoint {self.root} is already listening after stop; "
                           "stop the existing server or configure a different port before restarting")

    def _stop(self) -> None:
        if self.cfg.server.stop_command:
            subprocess.run(self.cfg.server.stop_command, capture_output=True, timeout=120)
        if self._proc is not None:
            proc = self._proc
            if os.name != "nt":
                # vLLM workers can outlive the API process, including after a
                # failed startup. The whole session belongs to this run.
                self._signal_group(proc.pid, signal.SIGTERM)
            elif proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            if os.name != "nt":
                # Also covers workers that outlive an already-exited parent.
                self._signal_group(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
        self._proc = None

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    async def restart(self) -> dict:
        command = self.command
        if not command:
            raise RuntimeError("server.cache_reset is 'restart' but server.start_command is not configured")
        t0 = time.monotonic()
        self.recorder.emit("server_stop")
        self._stop()
        await self._require_unused_endpoint()
        stopped = time.monotonic()
        if self._log:
            self._log.close()
        self._log = (self.run_dir / "server.log").open("ab")
        self.recorder.emit("server_start", command=command)
        self._proc = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT,
                                      start_new_session=os.name != "nt")
        ready_s = await self.wait_healthy(self.cfg.server.ready_timeout_s)
        self.recorder.emit("server_ready", ready_s=ready_s)
        return {"method": "restart", "command": command, "stop_s": stopped - t0, "startup_s": ready_s}

    async def reset_endpoint(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(self.root + "/reset_prefix_cache")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"/reset_prefix_cache failed: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"/reset_prefix_cache returned HTTP {resp.status_code}; it requires VLLM_SERVER_DEV_MODE=1")
        return {"method": "endpoint", "status": resp.status_code}

    async def warmup(self) -> dict:
        path = self.cfg.server.warmup_file
        if not path:
            return {"requests": 0}
        prompts = [json.loads(line) for line in self.cfg.path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        durations = []
        for i, item in enumerate(prompts):
            messages = [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}]
            ctx = RequestContext(self.recorder.run_id, "warmup", "warmup", f"warmup-{i}", i, 0)
            resp = await self.client.complete(messages, item.get("max_tokens", 32), ctx, [])
            durations.append(resp.duration_s)
        if self.cfg.workload.edit_mode == "whole_file":
            # Compile both fixed action schemas before the measured window.
            # These replies are only warm-up data; no generated file is executed.
            for first_write in (True, False):
                instruction = ('Reply with a write_file action for src/warmup.py containing pass and a newline.'
                               if first_write else 'Reply with {"action":"finish"}.')
                ctx = RequestContext(self.recorder.run_id, "warmup", "warmup", f"warmup-schema-{first_write}",
                                     len(durations), 0, whole_file_schema(first_write))
                resp = await self.client.complete([{"role": "user", "content": instruction}], 128, ctx, [])
                durations.append(resp.duration_s)
        return {"requests": len(durations), "durations_s": durations}

    async def snapshot(self) -> dict:
        text, error = await fetch_metrics(self.cfg.model.base_url)
        if text is None:
            return {"available": False, "error": error}
        return {"available": True, "signals": extract_signals(parse_prometheus(text))}

    async def wait_idle(self, timeout_s: float) -> dict:
        started = time.monotonic()
        while True:
            snap = await self.snapshot()
            if not snap["available"]:
                return {"verified": False, "reason": f"metrics unavailable: {snap.get('error')}"}
            running = snap["signals"]["requests_running"]
            waiting = snap["signals"]["requests_waiting"]
            if running is None or waiting is None:
                return {"verified": False, "reason": "request gauges not exposed by this server version"}
            if running == 0 and waiting == 0:
                return {"verified": True, "waited_s": time.monotonic() - started}
            if time.monotonic() - started > timeout_s:
                return {"verified": False, "reason": f"server still busy after {timeout_s}s",
                        "running": running, "waiting": waiting}
            await asyncio.sleep(0.5)

    async def version(self) -> dict | None:
        resp = await self._get("/version")
        if resp is not None and resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return {"raw": resp.text[:200]}
        return None

    async def prepare(self) -> dict:
        """Reset, warm up and confirm idle. Every step is timed as run overhead."""
        t0 = time.monotonic()
        method = self.cfg.server.cache_reset
        if method == "restart":
            reset = await self.restart()
        elif method == "endpoint":
            await self.wait_healthy(self.cfg.server.ready_timeout_s)
            reset = await self.reset_endpoint()
        else:
            await self.wait_healthy(self.cfg.server.ready_timeout_s)
            reset = {"method": "none", "warning": "prefix cache not reset; results are exploratory"}
        self.recorder.emit("cache_reset", **reset)
        warm = await self.warmup()
        self.recorder.emit("warmup_done", **warm)
        idle = await self.wait_idle(self.cfg.server.drain_timeout_s)
        self.recorder.emit("server_idle_before_run", **idle)
        if idle.get("verified") is not True:
            raise RuntimeError("server idleness before the cohort was not verified: "
                               f"{idle.get('reason', 'verification missing')}; refusing to start the measured workload")
        snap = await self.snapshot()
        return {
            "reset": reset,
            "warmup": warm,
            "idle_before_run": idle,
            "metrics_after_warmup": snap,
            "version": await self.version(),
            "overhead_s": time.monotonic() - t0,
        }

    def shutdown(self) -> dict:
        if self.cfg.server.cache_reset == "restart" and self.cfg.server.stop_after_run:
            self._stop()
            if self._log:
                self._log.close()
            return {"stopped": True}
        return {"stopped": False}
