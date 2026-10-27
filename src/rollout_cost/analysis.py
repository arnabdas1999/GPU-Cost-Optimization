"""Run summaries, baseline/candidate comparison and the acceptance rule.

Objective: batch completion. A configuration is better if the same fixed
cohort reaches terminal states using less allocated GPU time, without worse
outcomes. At a fixed hourly rate, steady-state cost is proportional to cohort
makespan, so a makespan guardrail would be redundant and none is applied. An
optional per-episode service-time guardrail exists for workloads where
individual responsiveness matters.
"""

from __future__ import annotations

import json
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import SUPPORTED_RUN_FORMATS
from .cost import ESTIMATED_COST_PREFIX, UNKNOWN_COST_WARNING, cost_notice, cost_summary
from .outcomes import CATEGORIES, GRADER_OUTCOMES, STOP_REASONS

DEFAULT_ACCEPTANCE: dict[str, Any] = {
    # Pooled over every paired trial: candidate successes may not drop by more than this.
    "max_pooled_success_drop": 0,
    # Per task, pooled over trials and repetitions: a drop in pass count larger
    # than this tolerance counts as a task regression.
    "task_regression_tolerance": 0,
    "max_task_regressions": 0,
    "max_infra_error_increase": 0,
    "max_wall_timeout_increase": 0,
    # Pooled allocated-window reduction required (cost is proportional at a fixed rate).
    "min_window_reduction_pct": 5.0,
    # Each paired trial must individually show a shorter window.
    "require_reduction_every_pair": True,
    # Noise floor: the pooled reduction must exceed the relative spread
    # (max - min) / mean of either side's windows across trials. An A/A run
    # (identical configs) on a development machine showed a 7.5% "reduction"
    # from noise alone, so a fixed threshold is not sufficient by itself.
    "min_trials": 2,
    "require_reduction_above_spread": True,
    # Optional responsiveness guardrail on admission-to-terminal episode time.
    "max_p90_service_time_increase_pct": None,
}

SMALL_BACKLOG_FACTOR = 3
LOW_TOOL_FRACTION = 0.10
HIGH_PROTOCOL_ERROR_RATE = 0.20


@dataclass
class Run:
    path: Path
    manifest: dict
    episodes: list[dict]
    summary: dict


def portable(path: str | Path) -> str:
    """A path as stored in artifacts: relative to the working directory when possible, always POSIX."""
    p = Path(path)
    try:
        p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return p.as_posix()


class RunFormatError(ValueError):
    pass


def load_run(path: str | Path) -> Run:
    path = Path(str(path).replace("\\", "/"))
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    version = manifest.get("format_version", 1)  # runs written before versioning are format 1
    if version not in SUPPORTED_RUN_FORMATS:
        raise RunFormatError(f"{path}: run format {version} is not supported (supported: {sorted(SUPPORTED_RUN_FORMATS)})")
    episodes = [json.loads(line) for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes.sort(key=lambda e: e["index"])
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return Run(path, manifest, episodes, summary)


def price_run(run: Run, cost_config: dict | None) -> Run:
    """Return a valuation for reporting/export; preserve measured artifacts and comparison inputs."""
    if cost_config is None:
        return run
    s = deepcopy(run.summary)
    if not s:
        raise RunFormatError(f"{run.path}: no summary.json (run incomplete or aborted)")
    s["cost"] = cost_summary(s["window_s"], s["run_inclusive_s"], s["valid_episodes"], s["successes"], cost_config)
    s["cost"]["post_run_valuation"] = True
    s["warnings"] = [w for w in s["warnings"]
                     if w != UNKNOWN_COST_WARNING and not w.startswith(ESTIMATED_COST_PREFIX)]
    notice = cost_notice(s["cost"])
    if notice:
        s["warnings"].append(notice)
    return Run(run.path, run.manifest, run.episodes, s)


def percentiles(values: list[float], points=(50, 90, 99)) -> dict[str, float | None]:
    data = sorted(v for v in values if v is not None)
    out: dict[str, float | None] = {"n": len(data)}
    for p in points:
        if not data:
            out[f"p{p}"] = None
            continue
        rank = (p / 100) * (len(data) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(data) - 1)
        out[f"p{p}"] = data[lo] + (data[hi] - data[lo]) * (rank - lo)
    out["max"] = data[-1] if data else None
    return out


def time_at_cap(intervals: list[tuple[float, float]], cap: int, start: float, end: float) -> float:
    """Fraction of [start, end] during which at least ``cap`` intervals overlap."""
    if end <= start or cap <= 0:
        return 0.0
    points = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda p: (p[0], p[1]))
    active, last, at_cap = 0, start, 0.0
    for t, delta in points:
        t = min(max(t, start), end)
        if active >= cap:
            at_cap += t - last
        last = t
        active += delta
    if active >= cap:
        at_cap += end - last
    return at_cap / (end - start)


def summarize_run(manifest: dict, episodes: list[dict], timeline: dict) -> dict:
    cfg = manifest["config"]
    n = len(episodes)
    categories = Counter(e["category"] for e in episodes)
    outcomes = {c: categories.get(c, 0) for c in CATEGORIES}
    valid = n - outcomes["infra_error"]
    successes = outcomes["success"]
    stop_reasons = Counter(e["stop_reason"] for e in episodes)
    grader = Counter(e["grader_outcome"] for e in episodes)
    crosstab = {s: {g: 0 for g in GRADER_OUTCOMES} for s in STOP_REASONS}
    for e in episodes:
        crosstab[e["stop_reason"]][e["grader_outcome"]] += 1

    start, end = timeline["window_start"], timeline["window_end"]
    window_s = max(0.0, end - start)
    run_inclusive_s = max(0.0, timeline["drain_end"] - timeline["prep_start"])

    cap = min(cfg["execution"]["max_active_episodes"], n) if n else 0
    intervals = [(e["admitted_t"], e["agent_end_t"]) for e in episodes]
    steady = time_at_cap(intervals, cap, start, end)

    sums = {k: sum(e[k] for e in episodes) for k in ("llm_wait_s", "request_s", "tool_wait_s", "tool_s", "file_op_s", "service_s")}
    service_total = sums["service_s"]
    attribution = {k: (sums[k] / service_total if service_total else None)
                   for k in ("llm_wait_s", "request_s", "tool_wait_s", "tool_s", "file_op_s")}
    if service_total:
        attribution["other_s"] = max(0.0, 1 - sum(v for v in attribution.values() if v is not None))

    steps = sum(e["steps"] for e in episodes)
    protocol_errors = Counter()
    for e in episodes:
        protocol_errors.update(e["protocol_errors"])
    prompt = [e["prompt_tokens"] for e in episodes]
    completion = [e["completion_tokens"] for e in episodes]
    usage_complete = all(v is not None for v in prompt + completion)

    prefix = prefix_cache_delta(timeline.get("server_prep", {}).get("metrics_after_warmup"), timeline.get("metrics_after_run"))

    summary = {
        "run_id": manifest["run_id"],
        "label": manifest.get("label"),
        "synthetic": manifest["synthetic"],
        "split": manifest["split"],
        "cohort_size": n,
        "tasks": len({e["task_id"] for e in episodes}),
        "repetitions": cfg["workload"]["repetitions"],
        "max_active_episodes": cfg["execution"]["max_active_episodes"],
        "max_inflight_llm_requests": cfg["execution"]["max_inflight_llm_requests"],
        "tool_workers": manifest["protected"]["tool_workers"],
        "outcomes": outcomes,
        "valid_episodes": valid,
        "successes": successes,
        "stop_reasons": dict(stop_reasons),
        "grader_outcomes": dict(grader),
        "stop_by_grade": crosstab,
        "successes_without_finish": sum(1 for e in episodes if e["category"] == "success" and e["stop_reason"] != "finished"),
        "wall_timeouts": stop_reasons.get("wall_timeout", 0),
        "window_s": window_s,
        "run_inclusive_s": run_inclusive_s,
        "overhead_s": timeline.get("server_prep", {}).get("overhead_s"),
        "cost": cost_summary(window_s, run_inclusive_s, valid, successes, cfg["cost"]),
        "steady_state_fraction": steady,
        "attribution": attribution,
        "tool_time_fraction": (sums["tool_s"] / service_total) if service_total else None,
        "latency": {
            "request_s": percentiles([d for e in episodes for d in e["request_durations_s"]]),
            "ttft_s": percentiles([d for e in episodes for d in e["ttft_s"]]),
            "llm_wait_per_request_s": percentiles([d for e in episodes for d in e["request_waits_s"]]),
            "service_s": percentiles([e["service_s"] for e in episodes]),
            "episode_latency_s": percentiles([e["latency_s"] for e in episodes]),
            "admission_wait_s": percentiles([e["admission_wait_s"] for e in episodes]),
        },
        "steps": steps,
        "protocol_errors": dict(protocol_errors),
        "protocol_error_rate": (sum(protocol_errors.values()) / steps) if steps else None,
        "tokens": {
            "usage_complete": usage_complete,
            "prompt_total": sum(prompt) if usage_complete else None,
            "completion_total": sum(completion) if usage_complete else None,
        },
        "cancelled_requests": sum(e["cancelled_requests"] for e in episodes),
        "max_inflight_observed": timeline.get("max_inflight_observed"),
        "max_tool_slots_observed": timeline.get("max_tool_slots_observed"),
        "prefix_cache": prefix,
        "drain": timeline.get("drain"),
        "telemetry": timeline.get("telemetry"),
    }
    summary["warnings"] = run_warnings(manifest, summary)
    return summary


def prefix_cache_delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after or not before.get("available") or not after.get("available"):
        return {"available": False}
    b, a = before["signals"], after["signals"]
    q0, q1, h0, h1 = b.get("prefix_cache_queries_total"), a.get("prefix_cache_queries_total"), b.get("prefix_cache_hits_total"), a.get("prefix_cache_hits_total")
    if None in (q0, q1, h0, h1):
        return {"available": False, "reason": "prefix-cache counters not exposed"}
    queries, hits = q1 - q0, h1 - h0
    return {"available": True, "queries": queries, "hits": hits, "hit_rate": hits / queries if queries else None}


def run_warnings(manifest: dict, s: dict) -> list[str]:
    cfg = manifest["config"]
    warnings: list[str] = []
    if manifest["synthetic"]:
        reason = manifest.get("synthetic_reason") or "scripted model client"
        warnings.append(f"SYNTHETIC: {reason}. Verifies software behavior only; not evidence of GPU efficiency.")
    cap = cfg["execution"]["max_active_episodes"]
    n = s["cohort_size"]
    if cap > n:
        warnings.append(f"max_active_episodes={cap} exceeds the cohort size {n}: the cap can never be reached.")
    elif n < SMALL_BACKLOG_FACTOR * cap:
        warnings.append(f"Cohort of {n} is under {SMALL_BACKLOG_FACTOR}x the episode cap {cap}: fill and drain dominate; "
                        "increase workload.repetitions for throughput comparisons.")
    if s["tool_time_fraction"] is not None and s["tool_time_fraction"] < LOW_TOOL_FRACTION:
        warnings.append(f"Tool execution is {s['tool_time_fraction']:.0%} of episode time: little room for model/tool overlap.")
    if s["protocol_error_rate"] is not None and s["protocol_error_rate"] > HIGH_PROTOCOL_ERROR_RATE:
        warnings.append(f"Protocol errors are {s['protocol_error_rate']:.0%} of replies: the run may mostly measure protocol failures.")
    if s["wall_timeouts"]:
        warnings.append(f"{s['wall_timeouts']} episode(s) hit the wall-time backstop; budgets were not scheduling-neutral.")
    if not s["tokens"]["usage_complete"]:
        warnings.append("Token usage was not reported for every request; token totals are unavailable.")
    drain = s.get("drain") or {}
    idle_before = (manifest.get("timeline", {}).get("server_prep") or {}).get("idle_before_run") or {}
    if not idle_before.get("verified", False) and not manifest["synthetic"]:
        warnings.append(f"Server idleness before the cohort was not verified: {idle_before.get('reason')}.")
    if not drain.get("server_idle", {}).get("verified", False) and not manifest["synthetic"]:
        warnings.append(f"Server idleness after the cohort was not verified: {drain.get('server_idle', {}).get('reason')}.")
    if drain.get("leftover_tool_processes"):
        warnings.append(f"Tool processes were still running after the cohort and were killed: {drain['leftover_tool_processes']}.")
    if cfg["server"]["cache_reset"] == "none" and not manifest["synthetic"]:
        warnings.append("Prefix cache was not reset before the run: results are exploratory.")
    notice = cost_notice(s["cost"])
    if notice:
        warnings.append(notice)
    return warnings


# ---------------------------------------------------------------------------
# Comparison


def outcome_bin(episode: dict) -> str:
    if episode["category"] == "infra_error":
        return "infra"
    return "pass" if episode["category"] == "success" else "fail"


def protected_diff(a: dict, b: dict) -> list[dict]:
    keys = sorted(set(a) | set(b))
    return [{"key": k, "baseline": a.get(k), "candidate": b.get(k)} for k in keys if a.get(k) != b.get(k)]


def run_problems(run: Run) -> list[str]:
    """Check the recorded cohort and decision inputs before comparing a run.

    Recompute decision values from episodes and the manifest timeline rather
    than trusting a summary that may belong to a different or incomplete run.
    Unverified real-server idleness leaves artifacts readable, but makes them
    ineligible for acceptance.
    """
    manifest, summary = run.manifest, run.summary
    problems: list[str] = []
    if manifest.get("status") != "complete":
        problems.append(f"run status is {manifest.get('status')!r}, not 'complete'")
    expected = manifest["protected"].get("cohort") or []
    actual = [e["episode_id"] for e in run.episodes]
    if not expected or len(expected) != len(set(expected)):
        problems.append("declared cohort is empty or contains duplicate episode IDs")
    if Counter(actual) != Counter(expected):
        problems.append("episode records must contain exactly one terminal record per declared episode "
                        "(missing, unexpected or duplicate records found)")
    try:
        derived = summarize_run(manifest, run.episodes, manifest["timeline"])
        stored_headline, derived_headline = _headline(summary), _headline(derived)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunFormatError(f"{run.path}: incomplete or invalid run artifacts: {exc}") from exc
    mismatches = [key for key in stored_headline if stored_headline[key] != derived_headline[key]]
    mismatches.extend(key for key in ("run_id", "synthetic", "outcomes") if summary.get(key) != derived[key])
    if mismatches:
        problems.append("summary disagrees with episode records or manifest: " + ", ".join(mismatches))
    if not manifest["synthetic"]:
        timeline = manifest["timeline"]
        idle_checks = (
            ("before the cohort", (timeline.get("server_prep") or {}).get("idle_before_run") or {}),
            ("after the cohort", (timeline.get("drain") or {}).get("server_idle") or {}),
        )
        for phase, idle in idle_checks:
            if idle.get("verified") is not True:
                problems.append(f"server idleness {phase} was not verified: "
                                f"{idle.get('reason', 'verification missing')}")
    return problems


def compare_runs(base: Run, cand: Run) -> dict:
    problems: list[str] = []
    for role, run in (("baseline", base), ("candidate", cand)):
        problems.extend(f"{role}: {problem}" for problem in run_problems(run))
    diff = protected_diff(base.manifest["protected"], cand.manifest["protected"])
    if diff:
        problems.append(f"{len(diff)} protected setting(s) differ: " + ", ".join(d["key"] for d in diff[:8]))
    if base.manifest["synthetic"] != cand.manifest["synthetic"]:
        problems.append("one run is synthetic and the other is not")
    b_eps = {e["episode_id"]: e for e in base.episodes}
    c_eps = {e["episode_id"]: e for e in cand.episodes}
    if set(b_eps) != set(c_eps):
        problems.append("the runs do not contain the same episodes")

    transitions: Counter[tuple[str, str]] = Counter()
    changed = []
    for eid in sorted(set(b_eps) & set(c_eps), key=lambda k: b_eps[k]["index"]):
        pair = (outcome_bin(b_eps[eid]), outcome_bin(c_eps[eid]))
        transitions[pair] += 1
        if pair[0] != pair[1]:
            changed.append({"episode_id": eid, "baseline": pair[0], "candidate": pair[1]})

    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"baseline": 0, "candidate": 0, "episodes": 0})
    for eid in set(b_eps) & set(c_eps):
        t = per_task[b_eps[eid]["task_id"]]
        t["episodes"] += 1
        t["baseline"] += outcome_bin(b_eps[eid]) == "pass"
        t["candidate"] += outcome_bin(c_eps[eid]) == "pass"

    bs, cs = base.summary, cand.summary
    return {
        "baseline": portable(base.path),
        "candidate": portable(cand.path),
        "baseline_run_id": base.manifest["run_id"],
        "candidate_run_id": cand.manifest["run_id"],
        "valid": not problems,
        "problems": problems,
        "protected_diff": diff,
        "varied": varied_settings(base.manifest["config"], cand.manifest["config"]),
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
        "changed_episodes": changed,
        "per_task": dict(sorted(per_task.items())),
        "baseline_summary": _headline(bs),
        "candidate_summary": _headline(cs),
        "cost_assumptions": {"baseline": bs["cost"], "candidate": cs["cost"]},
    }


def varied_settings(a: dict, b: dict) -> dict:
    from .config import TUNABLE_KEYS, flatten

    fa, fb = flatten(a), flatten(b)
    return {k: {"baseline": fa.get(k), "candidate": fb.get(k)} for k in sorted(TUNABLE_KEYS) if fa.get(k) != fb.get(k)}


def _headline(s: dict) -> dict:
    return {
        "window_s": s["window_s"],
        "cost": s["cost"]["window"]["cost"],
        "successes": s["successes"],
        "valid_episodes": s["valid_episodes"],
        "infra_errors": s["outcomes"]["infra_error"],
        "wall_timeouts": s["wall_timeouts"],
        "service_p90_s": s["latency"]["service_s"]["p90"],
        "cohort_size": s["cohort_size"],
    }


def evaluate_acceptance(comparisons: list[dict], rule: dict | None = None) -> dict:
    """Apply the predeclared rule to one or more paired trials. Every check is reported."""
    r = {**DEFAULT_ACCEPTANCE, **(rule or {})}
    unknown = set(rule or {}) - set(DEFAULT_ACCEPTANCE)
    if unknown:
        raise ValueError(f"unknown acceptance keys: {sorted(unknown)}")
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    invalid = [p for c in comparisons for p in c["problems"]]
    check("comparisons are valid", not invalid, "; ".join(invalid) or "protected settings and cohorts match")
    aa = not any(c["varied"] for c in comparisons)
    check("candidate differs from baseline", not aa,
          "no tunable setting differs: this is an A/A calibration; use its spread as the noise floor" if aa
          else ", ".join(sorted({k for c in comparisons for k in c["varied"]})))
    check("enough paired trials", len(comparisons) >= r["min_trials"], f"{len(comparisons)} trial(s), {r['min_trials']} required")

    b = [c["baseline_summary"] for c in comparisons]
    k = [c["candidate_summary"] for c in comparisons]
    b_succ, c_succ = sum(x["successes"] for x in b), sum(x["successes"] for x in k)
    check("pooled successes", b_succ - c_succ <= r["max_pooled_success_drop"],
          f"baseline {b_succ}, candidate {c_succ} (max drop {r['max_pooled_success_drop']})")

    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"baseline": 0, "candidate": 0})
    for c in comparisons:
        for task, t in c["per_task"].items():
            pooled[task]["baseline"] += t["baseline"]
            pooled[task]["candidate"] += t["candidate"]
    regressions = sorted(t for t, v in pooled.items() if v["baseline"] - v["candidate"] > r["task_regression_tolerance"])
    check("task-level regressions", len(regressions) <= r["max_task_regressions"],
          f"{len(regressions)} regressed task(s){': ' + ', '.join(regressions) if regressions else ''} "
          f"(tolerance {r['task_regression_tolerance']} per task, max {r['max_task_regressions']})")

    b_inf, c_inf = sum(x["infra_errors"] for x in b), sum(x["infra_errors"] for x in k)
    check("infrastructure errors", c_inf - b_inf <= r["max_infra_error_increase"], f"baseline {b_inf}, candidate {c_inf}")
    b_wt, c_wt = sum(x["wall_timeouts"] for x in b), sum(x["wall_timeouts"] for x in k)
    check("wall-time backstop hits", c_wt - b_wt <= r["max_wall_timeout_increase"], f"baseline {b_wt}, candidate {c_wt}")

    b_win, c_win = sum(x["window_s"] for x in b), sum(x["window_s"] for x in k)
    reduction = (1 - c_win / b_win) * 100 if b_win else 0.0
    check("pooled allocated-window reduction", reduction >= r["min_window_reduction_pct"],
          f"{reduction:.1f}% (baseline {b_win:.1f}s, candidate {c_win:.1f}s; required {r['min_window_reduction_pct']}%)")
    if r["require_reduction_every_pair"]:
        per_pair = [f"{(1 - y['window_s'] / x['window_s']) * 100:.1f}%" if x["window_s"] else "n/a" for x, y in zip(b, k)]
        check("reduction in every paired trial", all(y["window_s"] < x["window_s"] for x, y in zip(b, k)), ", ".join(per_pair))
    spread = max(relative_spread([x["window_s"] for x in b]), relative_spread([y["window_s"] for y in k]))
    if r["require_reduction_above_spread"]:
        if len(comparisons) < 2:
            check("reduction exceeds trial-to-trial spread", False, "spread unknown with a single trial")
        else:
            check("reduction exceeds trial-to-trial spread", reduction > spread,
                  f"reduction {reduction:.1f}% vs largest within-config spread {spread:.1f}%")
    limit = r["max_p90_service_time_increase_pct"]
    if limit is not None:
        worst = max(((y["service_p90_s"] / x["service_p90_s"] - 1) * 100) for x, y in zip(b, k) if x["service_p90_s"])
        check("p90 episode service time", worst <= limit, f"worst pair change {worst:+.1f}% (max {limit}%)")

    return {
        "accepted": all(c["ok"] for c in checks),
        "aa_calibration": aa,
        "window_spread_pct": spread if len(comparisons) >= 2 else None,
        "checks": checks,
        "rule": r,
        "pooled_per_task": dict(sorted(pooled.items())),
        "trials": len(comparisons),
        "note": "Prototype decision rule over a small sample; not a statistical test of equivalence.",
    }


def project_root(start: Path) -> Path:
    root = start.resolve()
    while not (root / "pyproject.toml").exists() and root.parent != root:
        root = root.parent
    return root


@dataclass
class Evaluated:
    """An experiment or sweep with its runs, comparisons and acceptance results."""

    name: str
    kind: str  # paired | sweep
    meta: dict
    baseline_runs: list[Run]
    candidate_runs: dict[str, list[Run]]  # candidate name -> runs, aligned with baseline_runs
    comparisons: dict[str, list[dict]]
    acceptance: dict[str, dict]
    synthetic: bool
    chosen: str | None = None


def evaluate_experiment_file(path: str | Path) -> Evaluated:
    """Paired experiment YAML written by `rollout-cost paired` (or by hand)."""
    import yaml

    path = Path(path)
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = project_root(path.parent)
    bases, cands, comps = [], [], []
    for pair in exp["pairs"]:
        base, cand = load_run(root / pair["baseline"]), load_run(root / pair["candidate"])
        bases.append(base)
        cands.append(cand)
        comps.append(compare_runs(base, cand))
    acceptance = evaluate_acceptance(comps, exp.get("acceptance"))
    synthetic = any(r.manifest["synthetic"] for r in [*bases, *cands])
    return Evaluated(exp["name"], "paired", {"split": exp.get("split"), "status": exp.get("status"), "source": portable(path)},
                     bases, {"candidate": cands}, {"candidate": comps}, {"candidate": acceptance}, synthetic)


def evaluate_sweep_selection(path: str | Path) -> Evaluated:
    """Sweep directory (or its selection.json) written by `rollout-cost sweep`."""
    path = Path(path)
    if path.is_dir():
        path = path / "selection.json"
    sel = json.loads(path.read_text(encoding="utf-8"))
    root = project_root(path.parent)
    runs = {name: load_run(root / p) for name, p in sel["runs"].items()}
    baseline_name = sel["baseline_variant"]
    base = runs[baseline_name]
    comps, accs, cands = {}, {}, {}
    for name, run in runs.items():
        if name == baseline_name:
            continue
        rule = ((sel.get("results") or {}).get(name) or {}).get("acceptance", {}).get("rule")
        comp = compare_runs(base, run)
        comps[name], cands[name] = [comp], [run]
        accs[name] = evaluate_acceptance([comp], rule or {"min_trials": 1, "require_reduction_above_spread": False})
    synthetic = any(r.manifest["synthetic"] for r in runs.values())
    return Evaluated(path.parent.name, "sweep", {"split": sel.get("split"), "status": "development selection",
                                                 "source": portable(path), "baseline_variant": baseline_name},
                     [base], cands, comps, accs, synthetic, chosen=sel.get("chosen"))


def verdict(acceptance: dict) -> str:
    if acceptance.get("aa_calibration"):
        return "A/A CALIBRATION"
    return "ACCEPTED" if acceptance["accepted"] else "NOT ACCEPTED"


def relative_spread(values: list[float]) -> float:
    """(max - min) / mean, in percent; 0 for fewer than two values."""
    if len(values) < 2 or not sum(values):
        return 0.0
    return (max(values) - min(values)) / (sum(values) / len(values)) * 100
