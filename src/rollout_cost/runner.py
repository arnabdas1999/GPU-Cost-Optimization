"""Run a fixed cohort of episodes under one configuration.

Lifecycle of a run (every phase is timestamped in run.json and events.jsonl):

1. Write the manifest (configuration, protected settings, hashes) before
   anything executes.
2. Server preparation: cache reset, fixed warm-up, idle check (overhead).
3. Measured window: from the first admission until every episode reaches a
   terminal, graded state. Admission is FIFO in cohort order through
   ``max_active_episodes`` workers; model requests are bounded separately.
4. Drain: kill any remaining tool processes and wait until the server reports
   no running or waiting requests, so nothing from this run leaks into the
   next one. The run is closed only after this.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import secrets
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .admission import FifoSemaphore
from .analysis import summarize_run
from .config import Config, fingerprint, protected_settings, render_command
from .contract import RUN_FORMAT_VERSION
from .fixtures import EpisodeSpec, build_cohort, load_split
from .grading import grade
from .harness import EpisodeEnv, EpisodeResult, run_episode
from .model_client import make_client
from .outcomes import derive_category
from .protocol import PROTOCOL_VERSION
from .report import render_run
from .server import ServerController
from .telemetry import EventRecorder, GpuSampler, ServerMetricsSampler, query_gpu_inventory, utc_now
from .tools import ToolPool, copy_tree, make_executor

# Modules whose code changes what a run does. Analysis/report/CLI are excluded so
# a report can be regenerated with newer code without invalidating comparisons.
BEHAVIOR_MODULES = ("admission.py", "config.py", "fixtures.py", "grading.py", "harness.py",
                    "model_client.py", "outcomes.py", "protocol.py", "runner.py", "server.py", "tools.py")


def new_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{name}-{secrets.token_hex(2)}"


def harness_identity() -> dict:
    import hashlib

    src = Path(__file__).parent
    digest = hashlib.sha256()
    for name in BEHAVIOR_MODULES:
        digest.update(name.encode())
        digest.update((src / name).read_bytes().replace(b"\r\n", b"\n"))
    commit = None
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=src, capture_output=True, text=True, timeout=10)
            commit += "-dirty" if dirty.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"version": __version__, "behavior_hash": digest.hexdigest(), "git_commit": commit}


def host_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "hostname": platform.node()}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        first = meminfo.read_text().splitlines()[0]
        info["mem_total_kb"] = int(first.split()[1])
    if hasattr(os, "sched_getaffinity"):
        info["runner_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return info


def build_manifest(cfg: Config, split: str, run_id: str, label: str | None, cohort: list[EpisodeSpec], executor, client) -> dict:
    cfg_dict = cfg.to_dict()
    tasks = {spec.task.task_id: spec.task for spec in cohort}
    harness = harness_identity()
    executor_info = executor.describe()
    protected = protected_settings(cfg_dict)
    protected.update({
        "split": split,
        "cohort": [s.episode_id for s in cohort],
        "fixtures": {t: task.fixture_hash for t, task in sorted(tasks.items())},
        "graders": {t: task.grader_hash for t, task in sorted(tasks.items())},
        "warmup_hash": _file_hash(cfg.path(cfg.server.warmup_file)) if cfg.server.warmup_file else None,
        "protocol_version": PROTOCOL_VERSION,
        "harness_behavior_hash": harness["behavior_hash"],
        "executor": executor_info["name"],
        "executor_image_id": executor_info.get("image_id"),
        "tool_workers": cfg.tool_worker_count,
    })
    return {
        "format_version": RUN_FORMAT_VERSION,
        "run_id": run_id,
        "label": label,
        "created_utc": utc_now(),
        "status": "started",
        "synthetic": cfg.synthetic,
        "synthetic_reason": "scripted model client" if cfg.synthetic else None,
        "split": split,
        "config": cfg_dict,
        "config_fingerprint": fingerprint(cfg_dict),
        "protected": protected,
        "protected_fingerprint": fingerprint(protected),
        "server_command": render_command(cfg.server.start_command, cfg.server.args) if cfg.server.start_command else None,
        "harness": harness,
        "host": host_info(),
        "gpu": query_gpu_inventory(cfg.telemetry.nvidia_smi),
        "executor": executor_info,
        "client": client.describe(),
    }


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class CohortRunner:
    def __init__(self, cfg: Config, split: str, run_dir: Path, run_id: str, label: str | None = None,
                 client=None, executor=None):
        self.cfg = cfg
        self.split = split
        self.run_dir = run_dir
        self.run_id = run_id
        self.label = label
        tasks = load_split(cfg, split)
        self.cohort = build_cohort(tasks, cfg.workload.repetitions, cfg.generation.seed)
        self.client = client or make_client(cfg, tasks)
        self.executor = executor or make_executor(cfg.tools.executor, cfg.tools.image, run_id,
                                                  cfg.tools.memory_mb, cfg.tools.pids_limit)
        self.records: list[dict] = []

    async def run(self) -> dict:
        cfg, run_dir = self.cfg, self.run_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        for sub in ("workspaces", "grading", "transcripts"):
            (run_dir / sub).mkdir()
        manifest = build_manifest(cfg, self.split, self.run_id, self.label, self.cohort, self.executor, self.client)
        # The runner shares the server's reserved CPUs, so tool-worker CPUs serve only tool work.
        previous_affinity = None
        if cfg.server.reserved_cpus and hasattr(os, "sched_setaffinity"):
            previous_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(cfg.server.reserved_cpus))
            manifest["host"]["runner_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        _write_json(run_dir / "run.json", manifest)
        try:
            return await self._run(manifest)
        finally:
            if previous_affinity is not None:
                os.sched_setaffinity(0, previous_affinity)

    async def _run(self, manifest: dict) -> dict:
        cfg, run_dir = self.cfg, self.run_dir

        recorder = EventRecorder(run_dir / "events.jsonl", self.run_id)
        llm_sem = FifoSemaphore(cfg.execution.max_inflight_llm_requests)
        pool = ToolPool(cfg.tools.worker_cpus, cfg.tools.workers)
        env = EpisodeEnv(self.run_id, cfg, self.client, llm_sem, pool, self.executor, recorder)
        server = None if cfg.synthetic else ServerController(cfg, run_dir, recorder, self.client)
        timeline: dict = {"prep_start": recorder.now(), "origin_utc": recorder.origin_utc}
        samplers = []
        gpu = GpuSampler(run_dir / "gpu_samples.jsonl", cfg.telemetry.gpu_sample_interval_s, recorder, cfg.telemetry.nvidia_smi)
        live = None
        if cfg.telemetry.live_metrics_port is not None:
            from .live import LiveExporter

            live = LiveExporter(cfg, manifest, recorder, llm_sem, pool, gpu)
            live.start()
        try:
            recorder.emit("run_start", cohort_size=len(self.cohort))
            timeline["server_prep"] = await server.prepare() if server else {"skipped": "synthetic run"}
            version = timeline["server_prep"].get("version") or {}
            if isinstance(version, dict) and version.get("mock"):
                manifest.update(synthetic=True, synthetic_reason="mock vLLM server (no GPU)")
            samplers.append(("gpu", gpu))
            if server:
                samplers.append(("server", ServerMetricsSampler(run_dir / "server_metrics.jsonl",
                                                                cfg.telemetry.server_metrics_interval_s, recorder, cfg.model.base_url)))
            for name, sampler in samplers:
                if name != "gpu" or gpu.available:
                    sampler.start()

            await self._run_cohort(env, recorder, timeline)

            recorder.emit("drain_start")
            drain = {"leftover_tool_processes": await self.executor.cleanup()}
            drain["server_idle"] = await server.wait_idle(cfg.server.drain_timeout_s) if server else {"verified": True, "reason": "synthetic"}
            timeline["drain"] = drain
            timeline["metrics_after_run"] = await server.snapshot() if server else None
            timeline["drain_end"] = recorder.now()
            recorder.emit("drain_end", **drain)
            status = "complete"
        except BaseException as exc:
            manifest["status"] = "aborted"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(run_dir / "run.json", manifest)
            await self.executor.cleanup()
            raise
        finally:
            for _, sampler in samplers:
                await sampler.stop()
            timeline["telemetry"] = {name: s.status() | ({"available": s.available} if name == "gpu" else {}) for name, s in samplers}
            if server:
                timeline["server_shutdown"] = server.shutdown()
            await self.client.aclose()
            recorder.emit("run_end")
            recorder.close()
            if live:
                live.stop()

        timeline["max_inflight_observed"] = llm_sem.max_in_use
        timeline["max_tool_slots_observed"] = pool.max_in_use
        manifest.update(status=status, finished_utc=utc_now(), timeline=timeline)
        _write_json(run_dir / "run.json", manifest)
        self.records.sort(key=lambda r: r["index"])
        summary = summarize_run(manifest, self.records, timeline)
        _write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(render_run(manifest, summary), encoding="utf-8")
        return summary

    async def _run_cohort(self, env: EpisodeEnv, recorder: EventRecorder, timeline: dict) -> None:
        pending = list(self.cohort)
        pending.reverse()  # pop() from the end yields cohort order
        grading: list[asyncio.Task] = []
        enqueued = recorder.now()
        timeline["window_start"] = enqueued
        recorder.emit("cohort_start", max_active_episodes=self.cfg.execution.max_active_episodes)
        episodes_fh = (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8")

        async def worker(worker_id: int) -> None:
            while pending:
                spec = pending.pop()
                admitted = recorder.now()
                recorder.emit("episode_admitted", spec.episode_id, worker=worker_id,
                              admission_wait_s=round(admitted - enqueued, 6))
                workspace = self.run_dir / "workspaces" / spec.episode_id
                try:
                    copy_tree(spec.task.workspace_dir, workspace)
                    result = await run_episode(env, spec, workspace)
                except Exception as exc:  # noqa: BLE001
                    result = EpisodeResult(stop_reason="infra_error", stop_detail="workspace setup failed",
                                           infrastructure_status="harness_error", infrastructure_detail=str(exc)[:500])
                agent_end = recorder.now()
                grading.append(asyncio.create_task(self._grade(env, spec, result, workspace, enqueued, admitted, agent_end, episodes_fh)))

        workers = [asyncio.create_task(worker(i)) for i in range(min(self.cfg.execution.max_active_episodes, len(self.cohort)))]
        try:
            await asyncio.gather(*workers)
            await asyncio.gather(*grading)
        finally:
            for task in [*workers, *grading]:
                task.cancel()
            episodes_fh.close()
        timeline["window_end"] = max((r["graded_t"] for r in self.records), default=enqueued)
        recorder.emit("cohort_end", window_s=timeline["window_end"] - timeline["window_start"])

    async def _grade(self, env: EpisodeEnv, spec: EpisodeSpec, result: EpisodeResult, workspace: Path,
                     enqueued: float, admitted: float, agent_end: float, fh) -> None:
        recorder = env.recorder
        recorder.emit("grading_start", spec.episode_id)
        if workspace.exists():
            graded = await grade(spec.task, workspace, self.run_dir / "grading" / spec.episode_id, self.executor,
                                 env.tool_pool, self.cfg.tools.grader_timeout_s, f"{spec.episode_id}-grade",
                                 on_slot=lambda slot: recorder.emit("grading_slot_acquired", spec.episode_id,
                                                                    slot=slot.index, cpu=slot.cpu))
            shutil.rmtree(self.run_dir / "grading" / spec.episode_id, ignore_errors=True)
            grade_dict = graded.to_dict()
        else:
            grade_dict = {"outcome": "not_graded", "detail": "no workspace", "exit_code": None, "duration_s": 0.0, "output_tail": ""}
        graded_t = recorder.now()
        category = derive_category(result.stop_reason, result.infrastructure_status, grade_dict["outcome"])
        record = {
            "index": spec.index,
            "episode_id": spec.episode_id,
            "task_id": spec.task.task_id,
            "family": spec.task.family,
            "rep": spec.rep,
            "seed": spec.seed,
            "fixture_hash": spec.task.fixture_hash,
            "grader_hash": spec.task.grader_hash,
            "category": category,
            **result.summary(),
            "grader_outcome": grade_dict["outcome"],
            "grader_detail": grade_dict["detail"],
            "grading_s": grade_dict["duration_s"],
            "enqueued_t": enqueued,
            "admitted_t": admitted,
            "agent_end_t": agent_end,
            "graded_t": graded_t,
            "admission_wait_s": admitted - enqueued,
            "service_s": agent_end - admitted,
            "latency_s": graded_t - enqueued,
            "request_durations_s": [q["duration_s"] for q in result.requests],
            "request_waits_s": [q["wait_s"] for q in result.requests],
        }
        transcript = {"episode_id": spec.episode_id, "messages": result.messages, "requests": result.requests,
                      "grade": grade_dict}
        _write_json(self.run_dir / "transcripts" / f"{spec.episode_id}.json", transcript)
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        self.records.append(record)
        recorder.emit("episode_terminal", spec.episode_id, status=category, stop_reason=result.stop_reason,
                      grader_outcome=grade_dict["outcome"], infrastructure=result.infrastructure_status)


async def execute_run(cfg: Config, split: str, runs_dir: Path, label: str | None = None, run_id: str | None = None,
                      client=None, executor=None) -> tuple[Path, dict]:
    run_id = run_id or new_run_id(cfg.name)
    run_dir = runs_dir / run_id
    runner = CohortRunner(cfg, split, run_dir, run_id, label, client=client, executor=executor)
    summary = await runner.run()
    return run_dir, summary


def run_sync(*args, **kwargs) -> tuple[Path, dict]:
    return asyncio.run(execute_run(*args, **kwargs))
