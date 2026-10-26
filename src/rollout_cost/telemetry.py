"""Event traces and background samplers for GPU and server metrics.

Times: ``t`` is monotonic seconds since the run's origin (for durations);
``utc`` is wall-clock ISO time (for joining with external logs). Events carry a
per-episode ``event_seq`` so ordering never depends on timestamp resolution.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

RUN_SCOPE = "__run__"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventRecorder:
    def __init__(self, path: Path, run_id: str, origin: float | None = None):
        self.path = path
        self.run_id = run_id
        self.origin = time.monotonic() if origin is None else origin
        # Wall-clock time of the monotonic origin: absolute time = origin_utc + t.
        self.origin_utc = (datetime.now(timezone.utc) - timedelta(seconds=time.monotonic() - self.origin)).isoformat()
        self._seq: dict[str, int] = defaultdict(int)
        self._fh = path.open("a", encoding="utf-8")
        # Observers (the live exporter) see every event after it is written.
        self.listeners: list = []

    def now(self) -> float:
        return time.monotonic() - self.origin

    def emit(self, event_type: str, episode_id: str | None = None, *, request_id: str | None = None,
             status: str | None = None, **data: Any) -> dict:
        scope = episode_id or RUN_SCOPE
        event = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "event_seq": self._seq[scope],
            "event_type": event_type,
            "request_id": request_id,
            "t": round(self.now(), 6),
            "utc": utc_now(),
            "status": status,
        }
        self._seq[scope] += 1
        if data:
            event["data"] = data
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._fh.flush()
        for listener in self.listeners:
            listener(event)
        return event

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Prometheus text parsing (vLLM /metrics)

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name across label sets. Comments are skipped."""
    totals: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        try:
            totals[match.group(1)] += float(match.group(3))
        except ValueError:
            continue
    return dict(totals)


# Names differ between vLLM releases; each signal lists known spellings and the
# report treats a signal with no match as unavailable rather than zero.
SERVER_SIGNALS: dict[str, tuple[str, ...]] = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemptions_total": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_cache_queries_total": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    "prefix_cache_hits_total": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
}


def extract_signals(metrics: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for signal, names in SERVER_SIGNALS.items():
        out[signal] = next((metrics[n] for n in names if n in metrics), None)
    return out


def server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


async def fetch_metrics(base_url: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """Return (raw text, error)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(server_root(base_url) + "/metrics")
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return resp.text, None
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Samplers


class _Sampler:
    def __init__(self, path: Path, interval_s: float, recorder: EventRecorder):
        self.path = path
        self.interval_s = interval_s
        self.recorder = recorder
        self.samples = 0
        self.last: dict | None = None
        self.errors: list[str] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def sample(self) -> dict | None:
        raise NotImplementedError

    async def _loop(self) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            while not self._stop.is_set():
                try:
                    row = await self.sample()
                except Exception as exc:  # noqa: BLE001 - telemetry must not break a run
                    row = None
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                if row is not None:
                    row = {"t": round(self.recorder.now(), 6), "utc": utc_now(), **row}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    self.samples += 1
                    self.last = row
                try:
                    await asyncio.wait_for(self._stop.wait(), self.interval_s)
                except TimeoutError:
                    pass

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def status(self) -> dict:
        return {"samples": self.samples, "errors": self.errors[:5], "error_count": len(self.errors)}


GPU_FIELDS = ("index", "utilization.gpu", "memory.used", "memory.total", "power.draw")


class GpuSampler(_Sampler):
    """Polls nvidia-smi. GPU utilization is diagnostic only, never the objective."""

    def __init__(self, path: Path, interval_s: float, recorder: EventRecorder, nvidia_smi: str = "nvidia-smi"):
        super().__init__(path, interval_s, recorder)
        self.binary = shutil.which(nvidia_smi)

    @property
    def available(self) -> bool:
        return self.binary is not None

    async def sample(self) -> dict | None:
        if not self.binary:
            return None
        proc = await asyncio.create_subprocess_exec(
            self.binary, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), 10)
        if proc.returncode != 0:
            raise RuntimeError(err.decode(errors="replace").strip())
        gpus = []
        for line in out.decode().strip().splitlines():
            values = [v.strip() for v in line.split(",")]
            gpus.append({k: _num(v) for k, v in zip(GPU_FIELDS, values)})
        return {"gpus": gpus}


class ServerMetricsSampler(_Sampler):
    def __init__(self, path: Path, interval_s: float, recorder: EventRecorder, base_url: str):
        super().__init__(path, interval_s, recorder)
        self.base_url = base_url

    async def sample(self) -> dict | None:
        text, error = await fetch_metrics(self.base_url)
        if text is None:
            raise RuntimeError(error)
        return {"signals": extract_signals(parse_prometheus(text))}


def _num(value: str) -> float | str | None:
    if value in ("[N/A]", "N/A", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def query_gpu_inventory(nvidia_smi: str = "nvidia-smi") -> dict:
    binary = shutil.which(nvidia_smi)
    if not binary:
        return {"available": False, "reason": f"{nvidia_smi} not found"}
    try:
        proc = subprocess.run(
            [binary, "--query-gpu=index,name,memory.total,driver_version,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        version = subprocess.run([binary], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", version)
    return {"available": proc.returncode == 0, "gpus": proc.stdout.strip().splitlines(),
            "cuda_driver_api": cuda.group(1) if cuda else None}
