"""Markdown rendering for runs, comparisons and experiments."""

from __future__ import annotations

from .outcomes import CATEGORIES, GRADER_OUTCOMES, STOP_REASONS


def _f(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _money(value, currency: str) -> str:
    return "unavailable" if value is None else f"{value:.4f} {currency}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_run(manifest: dict, s: dict) -> str:
    cfg = manifest["config"]
    cur = cfg["cost"]["currency"]
    lines = [f"# Run {s['run_id']}", ""]
    if s["synthetic"]:
        lines += ["> **SYNTHETIC RUN** — scripted model client. Timings exercise the software, not a GPU.", ""]
    for w in s["warnings"]:
        if not w.startswith("SYNTHETIC"):
            lines.append(f"- ⚠ {w}")
    if len(s["warnings"]) > int(s["synthetic"]):
        lines.append("")

    model = cfg["model"]
    gpu = manifest.get("gpu", {})
    lines += [
        "## Setup",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Model | {model['repository'] or model['served_name']} @ {model['revision'] or 'unpinned'} |",
        f"| GPU | {', '.join(gpu.get('gpus', [])) or 'not detected'} |",
        f"| Split / cohort | {s['split']}: {s['tasks']} tasks × {s['repetitions']} repetitions = {s['cohort_size']} episodes |",
        f"| Episode cap / request cap | {s['max_active_episodes']} / {s['max_inflight_llm_requests']} |",
        f"| Tool workers (fixed CPU budget) | {s['tool_workers']} ({manifest['executor']['name']}) |",
        f"| Cache procedure | {cfg['server']['cache_reset']} |",
        f"| Budgets | {cfg['workload']['max_steps']} steps, {cfg['generation']['max_output_tokens_per_call']} output tokens/call, "
        f"{cfg['workload']['context_window_tokens']} context, {cfg['workload']['episode_wall_timeout_s']:.0f}s wall backstop |",
        "",
        "## Outcomes",
        "",
        "| Category | Episodes |",
        "| --- | --- |",
    ]
    for c in CATEGORIES:
        lines.append(f"| {c} | {s['outcomes'][c]} |")
    lines += [
        f"| **submitted** | **{s['cohort_size']}** |",
        "",
        f"Valid evaluated episodes: {s['valid_episodes']} of {s['cohort_size']}. "
        f"Successes without a finish action: {s['successes_without_finish']}.",
        "",
        "Stop reason × grader outcome:",
        "",
        "| stop_reason | " + " | ".join(GRADER_OUTCOMES) + " |",
        "| --- |" + " --- |" * len(GRADER_OUTCOMES),
    ]
    for stop in STOP_REASONS:
        row = s["stop_by_grade"][stop]
        if any(row.values()):
            lines.append(f"| {stop} | " + " | ".join(str(row[g]) for g in GRADER_OUTCOMES) + " |")
    cost = s["cost"]
    lines += [
        "",
        "## Allocation time and cost",
        "",
        f"Rate: {_f(cost['window']['hourly_rate'])} {cur}/hour (source: {cost.get('rate_source') or 'not recorded'}).",
        "",
        "| Measure | Seconds | Billed seconds | Cost |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("window", "run_inclusive"):
        line = cost[key]
        lines.append(f"| {line['label']} | {line['seconds']:.1f} | {line['billed_seconds']:.1f} | {_money(line['cost'], cur)} |")
    lines += [
        "",
        f"Per valid episode: {_money(cost['per_valid_episode'], cur)} (n={s['valid_episodes']}). "
        f"Per success: {_money(cost['per_success'], cur) if s['successes'] else 'undefined (no successes)'} (n={s['successes']}).",
        "",
        "## Where episode time went",
        "",
        f"Episode slots were full for {_pct(s['steady_state_fraction'])} of the window; "
        f"tool execution was {_pct(s['tool_time_fraction'])} of episode service time.",
        "",
        "| Share of episode service time | |",
        "| --- | --- |",
    ]
    for k, v in s["attribution"].items():
        lines.append(f"| {k.removesuffix('_s')} | {_pct(v)} |")
    lines += ["", "| Latency (s) | n | p50 | p90 | p99 | max |", "| --- | --- | --- | --- | --- | --- |"]
    for name, p in s["latency"].items():
        lines.append(f"| {name.removesuffix('_s')} | {p['n']} | {_f(p['p50'])} | {_f(p['p90'])} | {_f(p['p99'])} | {_f(p['max'])} |")
    tok = s["tokens"]
    prefix = s["prefix_cache"]
    lines += [
        "",
        "## Protocol, tokens and server",
        "",
        f"- Replies: {s['steps']}; protocol errors: {sum(s['protocol_errors'].values())} {s['protocol_errors'] or ''}",
        f"- Prompt / completion tokens: {_f(tok['prompt_total'])} / {_f(tok['completion_total'])}"
        + ("" if tok["usage_complete"] else " (usage incomplete)"),
        f"- Cancelled requests: {s['cancelled_requests']}; max in-flight requests observed: {_f(s['max_inflight_observed'])}; "
        f"max tool slots in use: {_f(s['max_tool_slots_observed'])}",
        f"- Prefix-cache hit rate in window: {_pct(prefix.get('hit_rate')) if prefix.get('available') else 'unavailable'}",
        f"- Drain: {s.get('drain')}",
        "",
    ]
    return "\n".join(lines)


def render_comparison(comp: dict, acceptance: dict | None = None) -> str:
    b, c = comp["baseline_summary"], comp["candidate_summary"]
    lines = ["# Baseline vs candidate", "", f"- Baseline: `{comp['baseline']}`", f"- Candidate: `{comp['candidate']}`", ""]
    if not comp["valid"]:
        lines += ["> **INVALID COMPARISON**: " + "; ".join(comp["problems"]), ""]
    if comp["varied"]:
        lines += ["Varied settings:", ""] + [f"- `{k}`: {v['baseline']} → {v['candidate']}" for k, v in comp["varied"].items()] + [""]

    def delta(x, y, fmt="{:.1f}"):
        if x is None or y is None:
            return "n/a"
        pct = f" ({(y / x - 1) * 100:+.1f}%)" if x else ""
        return fmt.format(y - x) + pct

    lines += [
        "| Metric | Baseline | Candidate | Change |",
        "| --- | --- | --- | --- |",
        f"| Allocated window (s) | {b['window_s']:.1f} | {c['window_s']:.1f} | {delta(b['window_s'], c['window_s'])} |",
        f"| Window cost | {_f(b['cost'], 4)} | {_f(c['cost'], 4)} | {delta(b['cost'], c['cost'], '{:+.4f}')} |",
        f"| Successes / valid / submitted | {b['successes']} / {b['valid_episodes']} / {b['cohort_size']} | "
        f"{c['successes']} / {c['valid_episodes']} / {c['cohort_size']} | {c['successes'] - b['successes']:+d} |",
        f"| Infrastructure errors | {b['infra_errors']} | {c['infra_errors']} | {c['infra_errors'] - b['infra_errors']:+d} |",
        f"| Wall-time backstop hits | {b['wall_timeouts']} | {c['wall_timeouts']} | {c['wall_timeouts'] - b['wall_timeouts']:+d} |",
        f"| p90 episode service time (s) | {_f(b['service_p90_s'])} | {_f(c['service_p90_s'])} | {delta(b['service_p90_s'], c['service_p90_s'])} |",
        "",
        "Episode transitions (baseline → candidate, matched by task and repetition):",
        "",
        "| Transition | Episodes |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in comp["transitions"].items()]
    lines += ["", "| Task | Episodes | Baseline passes | Candidate passes |", "| --- | --- | --- | --- |"]
    lines += [f"| {t} | {v['episodes']} | {v['baseline']} | {v['candidate']} |" for t, v in comp["per_task"].items()]
    lines.append("")
    if acceptance:
        lines += render_acceptance(acceptance)
    return "\n".join(lines)


def render_acceptance(acc: dict) -> list[str]:
    if acc.get("aa_calibration"):
        spread = acc.get("window_spread_pct")
        verdict = f"A/A CALIBRATION (window spread {'n/a' if spread is None else f'{spread:.1f}%'})"
    else:
        verdict = "ACCEPTED" if acc["accepted"] else "NOT ACCEPTED"
    lines = [f"## Decision: {verdict}", "", f"Trials: {acc['trials']}. {acc['note']}", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {'pass' if c['ok'] else 'FAIL'} | {c['detail']} |" for c in acc["checks"]]
    lines.append("")
    return lines


def render_experiment(name: str, comparisons: list[dict], acceptance: dict, meta: dict) -> str:
    lines = [f"# Experiment: {name}", ""]
    if meta.get("status"):
        lines += [f"Status: **{meta['status']}**", ""]
    if meta.get("synthetic"):
        lines += ["> **SYNTHETIC** — runs used the scripted model client; no GPU-efficiency claim follows.", ""]
    lines += render_acceptance(acceptance)
    lines += ["## Pooled per-task passes", "", "| Task | Baseline | Candidate |", "| --- | --- | --- |"]
    lines += [f"| {t} | {v['baseline']} | {v['candidate']} |" for t, v in acceptance["pooled_per_task"].items()]
    lines.append("")
    for i, comp in enumerate(comparisons, 1):
        lines += [f"## Trial {i}", "", render_comparison(comp).split("\n", 1)[1]]
    return "\n".join(lines)
