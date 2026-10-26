"""Live Prometheus exporter for a run in progress (optional 'dashboard' extra).

Observation only: values are derived from the run's event stream (the same
events written to events.jsonl) and from read-only views of the admission
semaphore, tool pool and GPU sampler at scrape time. Nothing here can change
scheduling. Metric names come from ``contract.LIVE_METRICS``.
"""

from __future__ import annotations

from collections import Counter

from .contract import LIVE_LABELS, LIVE_METRICS


def _require_prometheus_client():
    try:
        import prometheus_client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise RuntimeError("telemetry.live_metrics_port is set but prometheus-client is not installed; "
                           "run `uv sync --extra dashboard`") from exc


class LiveState:
    """Run state maintained from recorder events."""

    def __init__(self) -> None:
        self.phase = "prep"
        self.active: set[str] = set()
        self.terminal: Counter[str] = Counter()
        self.requests: Counter[str] = Counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.window_start: float | None = None
        self.window_end: float | None = None

    def on_event(self, event: dict) -> None:
        kind = event["event_type"]
        data = event.get("data") or {}
        if kind == "cohort_start":
            self.phase, self.window_start = "cohort", event["t"]
        elif kind == "cohort_end":
            self.window_end = event["t"]
        elif kind == "drain_start":
            self.phase = "drain"
        elif kind == "run_end":
            self.phase = "done"
        elif kind == "episode_admitted":
            self.active.add(event["episode_id"])
        elif kind in ("episode_stop", "episode_terminal"):
            self.active.discard(event["episode_id"])
            if kind == "episode_terminal":
                self.terminal[event["status"]] += 1
        elif kind == "request_end":
            self.requests[event["status"] or "unknown"] += 1
            self.prompt_tokens += data.get("prompt_tokens") or 0
            self.completion_tokens += data.get("completion_tokens") or 0


class LiveCollector:
    def __init__(self, cfg, manifest: dict, recorder, llm_sem, pool, gpu, state: LiveState):
        self.cfg = cfg
        self.manifest = manifest
        self.recorder = recorder
        self.llm_sem = llm_sem
        self.pool = pool
        self.gpu = gpu
        self.state = state
        self.base = (manifest["run_id"], cfg.name)

    def _family(self, name: str):
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        spec = LIVE_METRICS[name]
        labels = [*LIVE_LABELS, *spec["labels"]]
        if spec["type"] == "counter":
            return CounterMetricFamily(name.removesuffix("_total"), spec["help"], labels=labels)
        return GaugeMetricFamily(name, spec["help"], labels=labels)

    def collect(self):
        s, base = self.state, self.base
        values: dict[str, list[tuple[tuple, float]]] = {
            "rc_run_info": [((str(self.manifest["synthetic"]).lower(), self.manifest["split"]), 1)],
            "rc_run_phase": [((phase,), float(phase == s.phase)) for phase in ("prep", "cohort", "drain", "done")],
            "rc_episodes_submitted": [((), len(self.manifest["protected"]["cohort"]))],
            "rc_episodes_active": [((), len(s.active))],
            "rc_episode_cap": [((), self.cfg.execution.max_active_episodes)],
            "rc_episodes_terminal_total": [((c,), n) for c, n in sorted(s.terminal.items())],
            "rc_llm_requests_inflight": [((), self.llm_sem.in_use)],
            "rc_llm_requests_waiting": [((), self.llm_sem.waiting)],
            "rc_llm_request_cap": [((), self.llm_sem.capacity)],
            "rc_llm_requests_total": [((st,), n) for st, n in sorted(s.requests.items())],
            "rc_tool_slots_busy": [((), self.pool.in_use)],
            "rc_tool_slots_waiting": [((), self.pool.waiting)],
            "rc_tool_slots_total": [((), self.pool.size)],
            "rc_prompt_tokens_total": [((), s.prompt_tokens)],
            "rc_completion_tokens_total": [((), s.completion_tokens)],
        }
        window = 0.0
        if s.window_start is not None:
            window = (s.window_end if s.window_end is not None else self.recorder.now()) - s.window_start
        values["rc_window_seconds"] = [((), window)]
        rate = self.cfg.cost.allocation_hourly_rate
        # Unknown price: the series is absent, never zero.
        values["rc_allocation_cost"] = [((self.cfg.cost.currency,), window / 3600 * rate)] if rate is not None else []
        gpus = (self.gpu.last or {}).get("gpus", []) if self.gpu is not None else []
        for metric, field in (("rc_gpu_utilization_percent", "utilization.gpu"), ("rc_gpu_memory_used_mib", "memory.used"),
                              ("rc_gpu_memory_total_mib", "memory.total")):
            values[metric] = [((str(int(g.get("index") or 0)),), g[field]) for g in gpus if isinstance(g.get(field), float)]
        for name, samples in values.items():
            family = self._family(name)
            for extra, value in samples:
                family.add_metric([*base, *extra], value)
            yield family


class LiveExporter:
    def __init__(self, cfg, manifest: dict, recorder, llm_sem, pool, gpu):
        _require_prometheus_client()
        self.cfg = cfg
        self.manifest = manifest
        self.state = LiveState()
        recorder.listeners.append(self.state.on_event)
        self.collector = LiveCollector(cfg, manifest, recorder, llm_sem, pool, gpu, self.state)
        self._server = None

    def start(self) -> None:
        from prometheus_client import CollectorRegistry, start_http_server

        registry = CollectorRegistry(auto_describe=False)
        registry.register(self.collector)
        host, port = self.cfg.telemetry.live_metrics_host, self.cfg.telemetry.live_metrics_port
        try:
            self._server, _ = start_http_server(port, addr=host, registry=registry)
            self.manifest["live_metrics"] = {"endpoint": f"http://{host}:{port}/metrics"}
        except OSError as exc:
            # The exporter is observational: a busy port must not fail the run, but it is recorded.
            self.manifest["live_metrics"] = {"error": f"could not bind {host}:{port}: {exc}"}

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
