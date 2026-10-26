"""Export run directories, paired experiments and sweeps into PostgreSQL for Grafana.

Row building is pure (``run_rows``, ``experiment_rows``) so it is tested without
a database. ``export`` replaces every row of a run or experiment in one
transaction, so exporting again is always safe. Values come from the same
summary, comparison and acceptance code as the Markdown reports.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

from .analysis import Evaluated, Run, evaluate_experiment_file, evaluate_sweep_selection, load_run, verdict
from .contract import EXPORT_SCHEMA_VERSION
from .spans import build_spans, concurrency, episode_states


class ExportError(RuntimeError):
    pass


def schema_sql() -> str:
    return resources.files("rollout_cost").joinpath("schema.sql").read_text(encoding="utf-8")


def schema_tables() -> dict[str, list[str]]:
    """Table -> columns, parsed from schema.sql (used by export and dashboard tests)."""
    tables: dict[str, list[str]] = {}
    for match in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\);", schema_sql(), re.S):
        columns = []
        for line in match.group(2).splitlines():
            line = line.split("--")[0].strip()
            if line:
                columns.append(line.split()[0])
        tables[match.group(1)] = columns
    return tables


RUN_TABLES = ("runs", "run_warnings", "episodes", "spans", "episode_states", "concurrency", "requests",
              "gpu_samples", "server_samples", "attribution", "latency")
EXPERIMENT_TABLES = ("experiments", "experiment_trials", "acceptance_checks", "transitions", "task_passes")


# ---------------------------------------------------------------------------
# Row building (pure)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _origin(manifest: dict, events: list[dict]) -> datetime:
    origin = (manifest.get("timeline") or {}).get("origin_utc")
    if origin:
        return datetime.fromisoformat(origin)
    first = events[0]
    return datetime.fromisoformat(first["utc"]) - timedelta(seconds=first["t"])


def run_rows(run: Run) -> dict[str, list[dict]]:
    m, s = run.manifest, run.summary
    if not s:
        raise ExportError(f"{run.path}: no summary.json (run incomplete or aborted)")
    events = _read_jsonl(run.path / "events.jsonl")
    origin = _origin(m, events)
    timeline = m.get("timeline") or {}

    def at(t: float | None) -> datetime | None:
        return None if t is None else origin + timedelta(seconds=t)

    cfg = m["config"]
    start_at = at(timeline.get("prep_start", 0.0))
    end_at = at(timeline.get("drain_end", timeline.get("window_end", 0.0)))
    cost = s["cost"]
    tokens = s["tokens"]
    rows: dict[str, list[dict]] = {name: [] for name in RUN_TABLES}
    rows["runs"].append({
        "run_id": m["run_id"], "label": m.get("label"), "config_name": cfg.get("name"), "split": m["split"],
        "synthetic": bool(m["synthetic"]), "synthetic_reason": m.get("synthetic_reason"), "status": m.get("status"),
        "format_version": m.get("format_version", 1), "created_utc": datetime.fromisoformat(m["created_utc"]),
        "start_at": start_at, "window_start": at(timeline.get("window_start")), "window_end": at(timeline.get("window_end")),
        "end_at": end_at, "start_ms": int(start_at.timestamp() * 1000), "end_ms": int(end_at.timestamp() * 1000) + 1000,
        "model": cfg["model"].get("repository") or cfg["model"].get("served_name"),
        "model_revision": cfg["model"].get("revision"),
        "gpu": ", ".join((m.get("gpu") or {}).get("gpus") or []) or None,
        "executor": (m.get("executor") or {}).get("name"), "cache_reset": cfg["server"]["cache_reset"],
        "max_active_episodes": s["max_active_episodes"], "max_inflight_llm_requests": s["max_inflight_llm_requests"],
        "tool_workers": s["tool_workers"], "tasks": s["tasks"], "repetitions": s["repetitions"],
        "cohort_size": s["cohort_size"], "valid_episodes": s["valid_episodes"], "successes": s["successes"],
        "failed_after_finish": s["outcomes"]["failed_after_finish"],
        "failed_budget_exhausted": s["outcomes"]["failed_budget_exhausted"],
        "infra_errors": s["outcomes"]["infra_error"], "wall_timeouts": s["wall_timeouts"],
        "window_s": s["window_s"], "run_inclusive_s": s["run_inclusive_s"], "overhead_s": s.get("overhead_s"),
        "hourly_rate": cost["window"]["hourly_rate"], "currency": cost["window"]["currency"],
        "window_cost": cost["window"]["cost"], "run_inclusive_cost": cost["run_inclusive"]["cost"],
        "cost_per_valid": cost["per_valid_episode"], "cost_per_success": cost["per_success"],
        "steady_state_fraction": s["steady_state_fraction"], "tool_time_fraction": s["tool_time_fraction"],
        "protocol_error_rate": s["protocol_error_rate"], "prompt_tokens": tokens["prompt_total"],
        "completion_tokens": tokens["completion_total"],
        "prefix_cache_hit_rate": (s.get("prefix_cache") or {}).get("hit_rate"),
        "cancelled_requests": s["cancelled_requests"],
        "drain_verified": ((s.get("drain") or {}).get("server_idle") or {}).get("verified"),
        "warnings_count": len(s["warnings"]), "config_fingerprint": m.get("config_fingerprint"),
        "protected_fingerprint": m.get("protected_fingerprint"),
    })
    rid = m["run_id"]
    rows["run_warnings"] = [{"run_id": rid, "ordinal": i, "message": w} for i, w in enumerate(s["warnings"])]
    for e in run.episodes:
        rows["episodes"].append({
            "run_id": rid, "episode_id": e["episode_id"], "idx": e["index"], "task_id": e["task_id"],
            "family": e.get("family"), "rep": e["rep"], "category": e["category"], "stop_reason": e["stop_reason"],
            "stop_detail": e.get("stop_detail"), "infrastructure_status": e["infrastructure_status"],
            "infrastructure_detail": e.get("infrastructure_detail"), "grader_outcome": e["grader_outcome"],
            "grader_detail": e.get("grader_detail"), "steps": e["steps"],
            "protocol_errors": sum((e.get("protocol_errors") or {}).values()), "tool_errors": e.get("tool_errors"),
            "prompt_tokens": e.get("prompt_tokens"), "completion_tokens": e.get("completion_tokens"),
            "llm_wait_s": e["llm_wait_s"], "request_s": e["request_s"], "tool_wait_s": e["tool_wait_s"],
            "tool_s": e["tool_s"], "file_op_s": e["file_op_s"], "grading_s": e.get("grading_s"),
            "service_s": e["service_s"], "latency_s": e["latency_s"], "admission_wait_s": e["admission_wait_s"],
            "admitted_at": at(e["admitted_t"]), "agent_end_at": at(e["agent_end_t"]), "graded_at": at(e["graded_t"]),
        })
    spans = build_spans(events)
    rows["spans"] = [{"run_id": rid, "episode_id": sp.episode_id, "kind": sp.kind, "start_at": at(sp.start),
                      "end_at": at(sp.end), "duration_s": sp.duration, "request_id": sp.request_id, "detail": sp.detail}
                     for sp in spans]
    rows["episode_states"] = [{"run_id": rid, "episode_id": eid, "ts": at(t), "state": code}
                              for eid, t, code in episode_states(events, spans)]
    for series, points in concurrency(spans).items():
        rows["concurrency"] += [{"run_id": rid, "ts": at(t), "series": series, "value": float(v)} for t, v in points]
    for path in sorted((run.path / "transcripts").glob("*.json")):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        for q in transcript.get("requests", []):
            rows["requests"].append({
                "run_id": rid, "episode_id": transcript["episode_id"], "request_id": q["request_id"], "step": q.get("step"),
                "status": q.get("status"), "duration_s": q.get("duration_s"), "wait_s": q.get("wait_s"),
                "ttft_s": q.get("ttft_s"), "prompt_tokens": q.get("prompt_tokens"),
                "completion_tokens": q.get("completion_tokens"), "finish_reason": q.get("finish_reason"),
                "attempts": len(q.get("attempts") or []),
            })
    def num(gpu: dict, key: str) -> float | None:
        value = gpu.get(key)
        return value if isinstance(value, (int, float)) else None

    for sample in _read_jsonl(run.path / "gpu_samples.jsonl"):
        for g in sample.get("gpus", []):
            rows["gpu_samples"].append({"run_id": rid, "ts": datetime.fromisoformat(sample["utc"]),
                                        "gpu": int(num(g, "index") or 0), "utilization": num(g, "utilization.gpu"),
                                        "memory_used": num(g, "memory.used"), "memory_total": num(g, "memory.total"),
                                        "power_draw": num(g, "power.draw")})
    for sample in _read_jsonl(run.path / "server_metrics.jsonl"):
        for signal, value in (sample.get("signals") or {}).items():
            if value is not None:
                rows["server_samples"].append({"run_id": rid, "ts": datetime.fromisoformat(sample["utc"]),
                                               "signal": signal, "value": float(value)})
    rows["attribution"] = [{"run_id": rid, "component": k.removesuffix("_s"), "share": v}
                           for k, v in (s.get("attribution") or {}).items()]
    rows["latency"] = [{"run_id": rid, "metric": k.removesuffix("_s"), "n": p["n"], "p50": p["p50"], "p90": p["p90"],
                        "p99": p["p99"], "max": p["max"]} for k, p in s["latency"].items()]
    return rows


def _trial_row(experiment: str, ordinal: int, x_label: str, role: str, candidate: str, run: Run) -> dict:
    s = run.summary
    return {"experiment": experiment, "ordinal": ordinal, "x_label": x_label, "role": role, "candidate": candidate,
            "run_id": run.manifest["run_id"], "max_active_episodes": s["max_active_episodes"],
            "max_inflight_llm_requests": s["max_inflight_llm_requests"], "window_s": s["window_s"],
            "window_cost": s["cost"]["window"]["cost"], "successes": s["successes"],
            "valid_episodes": s["valid_episodes"], "cohort_size": s["cohort_size"],
            "infra_errors": s["outcomes"]["infra_error"]}


def experiment_rows(ev: Evaluated) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name in EXPERIMENT_TABLES}
    name = ev.name
    if ev.kind == "paired":
        acc = ev.acceptance["candidate"]
        rows["experiments"].append({"name": name, "kind": "paired", "split": ev.meta.get("split"),
                                    "status": ev.meta.get("status"), "synthetic": ev.synthetic, "verdict": verdict(acc),
                                    "accepted": acc["accepted"], "aa_calibration": acc["aa_calibration"],
                                    "trials": acc["trials"], "window_spread_pct": acc.get("window_spread_pct"),
                                    "chosen": None, "source_path": ev.meta.get("source")})
        for i, (base, cand) in enumerate(zip(ev.baseline_runs, ev.candidate_runs["candidate"])):
            rows["experiment_trials"] += [_trial_row(name, 2 * i, f"trial {i + 1}", "baseline", "candidate", base),
                                          _trial_row(name, 2 * i + 1, f"trial {i + 1}", "candidate", "candidate", cand)]
    else:
        base = ev.baseline_runs[0]
        baseline_variant = ev.meta["baseline_variant"]
        rows["experiments"].append({"name": name, "kind": "sweep", "split": ev.meta.get("split"),
                                    "status": ev.meta.get("status"), "synthetic": ev.synthetic,
                                    "verdict": f"CHOSEN: {ev.chosen}", "accepted": ev.chosen != baseline_variant,
                                    "aa_calibration": False, "trials": 1, "window_spread_pct": None,
                                    "chosen": ev.chosen, "source_path": ev.meta.get("source")})
        variants = [(baseline_variant, "baseline", base)] + [(v, "candidate", runs[0]) for v, runs in ev.candidate_runs.items()]
        variants.sort(key=lambda item: (item[2].summary["max_active_episodes"], item[2].summary["max_inflight_llm_requests"]))
        rows["experiment_trials"] = [_trial_row(name, i, v, role, v, run) for i, (v, role, run) in enumerate(variants)]
    for candidate, acc in ev.acceptance.items():
        rows["acceptance_checks"] += [{"experiment": name, "candidate": candidate, "ordinal": i, "check_name": c["check"],
                                       "ok": c["ok"], "detail": c["detail"]} for i, c in enumerate(acc["checks"])]
        rows["task_passes"] += [{"experiment": name, "candidate": candidate, "task_id": task,
                                 "baseline_passes": v["baseline"], "candidate_passes": v["candidate"]}
                                for task, v in acc["pooled_per_task"].items()]
        for trial, comp in enumerate(ev.comparisons[candidate], 1):
            rows["transitions"] += [{"experiment": name, "candidate": candidate, "trial": trial, "transition": t,
                                     "episodes": n} for t, n in comp["transitions"].items()]
    return rows


def discover(runs_dir: Path, experiments_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    runs = sorted(p for p in runs_dir.iterdir() if (p / "run.json").exists()) if runs_dir.exists() else []
    paired = sorted(experiments_dir.glob("*.yaml")) if experiments_dir.exists() else []
    sweeps = sorted(p.parent for p in experiments_dir.glob("*/selection.json")) if experiments_dir.exists() else []
    return runs, paired, sweeps


# ---------------------------------------------------------------------------
# Database


def _connect(db_url: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise ExportError("psycopg is not installed; run `uv sync --extra dashboard`") from exc
    return psycopg.connect(db_url)


def ensure_schema(conn, reset: bool = False) -> None:
    tables = schema_tables()
    with conn.cursor() as cur:
        if reset:
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute(schema_sql())
        cur.execute("SELECT version FROM schema_version")
        found = [r[0] for r in cur.fetchall()]
        if not found:
            cur.execute("INSERT INTO schema_version (version) VALUES (%s)", (EXPORT_SCHEMA_VERSION,))
        elif found != [EXPORT_SCHEMA_VERSION]:
            raise ExportError(f"database schema version {found} does not match {EXPORT_SCHEMA_VERSION}; "
                              "re-run export with --reset to recreate the tables")
    conn.commit()


def _replace(cur, rows: dict[str, list[dict]], key_column: dict[str, str], key: str) -> int:
    count = 0
    for table, items in rows.items():
        cur.execute(f"DELETE FROM {table} WHERE {key_column[table]} = %s", (key,))
        if not items:
            continue
        columns = list(items[0])
        placeholders = ", ".join(["%s"] * len(columns))
        cur.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                        [tuple(item[c] for c in columns) for item in items])
        count += len(items)
    return count


def export(db_url: str, runs_dir: Path, experiments_dir: Path, reset: bool = False) -> dict:
    run_paths, paired, sweeps = discover(runs_dir, experiments_dir)
    report = {"runs": 0, "experiments": 0, "rows": 0, "skipped": []}
    run_keys = {t: "run_id" for t in RUN_TABLES}
    exp_keys = {t: "experiment" for t in EXPERIMENT_TABLES} | {"experiments": "name"}
    with _connect(db_url) as conn:
        ensure_schema(conn, reset)
        for path in run_paths:
            try:
                rows = run_rows(load_run(path))
            except (ExportError, KeyError, ValueError) as exc:
                report["skipped"].append(f"{path.name}: {exc}")
                continue
            with conn.transaction(), conn.cursor() as cur:
                report["rows"] += _replace(cur, rows, run_keys, rows["runs"][0]["run_id"])
            report["runs"] += 1
        for path, evaluate in [*((p, evaluate_experiment_file) for p in paired), *((p, evaluate_sweep_selection) for p in sweeps)]:
            try:
                rows = experiment_rows(evaluate(path))
            except (OSError, KeyError, ValueError) as exc:
                report["skipped"].append(f"{path.name}: {exc}")
                continue
            with conn.transaction(), conn.cursor() as cur:
                report["rows"] += _replace(cur, rows, exp_keys, rows["experiments"][0]["name"])
            report["experiments"] += 1
    return report
