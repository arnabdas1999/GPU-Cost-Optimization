"""Command-line interface: rollout-cost <command>."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from .analysis import DEFAULT_ACCEPTANCE, compare_runs, evaluate_acceptance, load_run
from .config import TUNABLE_KEYS, Config, ConfigError, flatten, load_config
from .fixtures import build_cohort, load_split
from .grading import validate_task
from .model_client import RequestContext, make_client
from .report import render_comparison, render_experiment, render_run
from .runner import run_sync
from .telemetry import extract_signals, parse_prometheus, query_gpu_inventory, server_root
from .tools import ToolPool, make_executor

MAX_SWEEP_CANDIDATES = 4


def _print_check(status: str, name: str, detail: str = "") -> None:
    print(f"  [{status:^4}] {name}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# validate-fixtures


def cmd_validate_fixtures(args) -> int:
    cfg = load_config(args.config)
    tasks = load_split(cfg, args.split)
    executor = make_executor(cfg.tools.executor, cfg.tools.image, "validate", cfg.tools.memory_mb, cfg.tools.pids_limit)

    async def go():
        pool = ToolPool(cfg.tools.worker_cpus, cfg.tools.workers)
        with tempfile.TemporaryDirectory(prefix="rc-validate-") as tmp:
            return await asyncio.gather(*(validate_task(t, Path(tmp), executor, pool, cfg.tools.grader_timeout_s) for t in tasks))

    results = asyncio.run(go())
    print(f"Fixture validation ({cfg.tools.executor} executor): starting code must fail, reference fix must pass")
    for r in results:
        _print_check("ok" if r["ok"] else "FAIL", r["task_id"],
                     f"starting public={r['starting_public']} hidden={r['starting_hidden']}; "
                     f"reference public={r['reference_public']} hidden={r['reference_hidden']}")
    return 0 if all(r["ok"] for r in results) else 1


# ---------------------------------------------------------------------------
# doctor


def cmd_doctor(args) -> int:
    failures = 0
    print(f"Checking {args.config}")
    try:
        cfg = load_config(args.config)
        warnings = cfg.validate()
        _print_check("ok", "configuration", cfg.name)
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        _print_check("FAIL", "configuration", str(exc))
        return 1
    for w in warnings:
        _print_check("warn", "config", w)
    placeholders = sorted(k for k, v in flatten(cfg.to_dict()).items()
                          if (isinstance(v, str) and v.startswith("REPLACE")) or (isinstance(v, list) and any(str(x).startswith("REPLACE") for x in v)))
    if placeholders and not cfg.synthetic:
        _print_check("FAIL", "pinned values", f"placeholders still set: {placeholders}")
        failures += 1

    try:
        tasks = load_split(cfg, args.split)
        cohort = build_cohort(tasks, cfg.workload.repetitions, cfg.generation.seed)
        _print_check("ok", f"split {args.split}", f"{len(tasks)} tasks x {cfg.workload.repetitions} reps = {len(cohort)} episodes")
        cap = cfg.execution.max_active_episodes
        if cap > len(cohort):
            _print_check("warn", "backlog", f"episode cap {cap} > cohort {len(cohort)}: the cap can never be reached")
        elif len(cohort) < 3 * cap:
            _print_check("warn", "backlog", f"cohort {len(cohort)} < 3x episode cap {cap}: fill/drain will dominate")
    except (ConfigError, OSError) as exc:
        _print_check("FAIL", "fixtures", str(exc))
        failures += 1

    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    missing = sorted(set(cfg.tools.worker_cpus + cfg.server.reserved_cpus) - set(cpus))
    if missing:
        _print_check("FAIL", "cpu sets", f"CPUs {missing} are not available to this process (have {cpus})")
        failures += 1
    else:
        _print_check("ok", "cpu sets", f"tool workers {cfg.tools.worker_cpus or f'{cfg.tools.workers} unpinned'}; "
                                        f"server reserved {cfg.server.reserved_cpus or 'none'}")

    if cfg.tools.executor == "docker":
        docker = shutil.which("docker")
        info = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True) if docker else None
        if not info or info.returncode != 0:
            _print_check("FAIL", "docker daemon", (info.stderr.strip()[:200] if info else "docker not found"))
            failures += 1
        else:
            image = subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", cfg.tools.image], capture_output=True, text=True)
            if image.returncode != 0:
                _print_check("FAIL", "tool image", f"{cfg.tools.image} not found; build docker/tool-runner.Dockerfile")
                failures += 1
            else:
                _print_check("ok", "tool image", f"{cfg.tools.image} {image.stdout.strip()[:19]}")
    else:
        _print_check("warn", "executor", "local executor: no isolation (development only)")

    gpu = query_gpu_inventory(cfg.telemetry.nvidia_smi)
    _print_check("ok" if gpu["available"] else "warn", "gpu", "; ".join(gpu.get("gpus", [])) or gpu.get("reason", ""))

    if cfg.synthetic:
        _print_check("ok", "model", f"scripted policy '{cfg.model.scripted_policy}' (synthetic)")
    else:
        failures += asyncio.run(_doctor_server(cfg))
    print("doctor: " + ("all required checks passed" if not failures else f"{failures} required check(s) failed"))
    return 0 if not failures else 1


async def _doctor_server(cfg: Config) -> int:
    failures = 0
    root = server_root(cfg.model.base_url)
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            models = (await http.get(cfg.model.base_url.rstrip("/") + "/models")).json()
            names = [m.get("id") for m in models.get("data", [])]
            ok = cfg.model.served_name in names
            _print_check("ok" if ok else "FAIL", "served model", f"{cfg.model.served_name} in {names}")
            failures += not ok
        except (httpx.HTTPError, ValueError) as exc:
            _print_check("FAIL", "model endpoint", f"{cfg.model.base_url}: {exc}")
            return 1
        try:
            resp = await http.get(root + "/metrics")
            signals = extract_signals(parse_prometheus(resp.text)) if resp.status_code == 200 else {}
            present = sorted(k for k, v in signals.items() if v is not None)
            absent = sorted(k for k, v in signals.items() if v is None)
            _print_check("ok" if present else "warn", "server metrics", f"present={present} absent={absent}")
        except httpx.HTTPError as exc:
            _print_check("warn", "server metrics", str(exc))
    client = make_client(cfg, [])
    try:
        ctx = RequestContext("doctor", "doctor", "doctor", "doctor-0", 0, cfg.generation.seed)
        resp = await client.complete([{"role": "user", "content": 'Reply with {"action": "finish"}'}], 16, ctx, [])
        usage_ok = resp.prompt_tokens is not None and resp.completion_tokens is not None
        _print_check("ok" if usage_ok else "warn", "streaming request",
                     f"ttft={resp.ttft_s and round(resp.ttft_s, 3)}s total={resp.duration_s:.3f}s usage={'reported' if usage_ok else 'missing'}")
    except Exception as exc:  # noqa: BLE001
        _print_check("FAIL", "streaming request", str(exc)[:300])
        failures += 1
    finally:
        await client.aclose()
    return failures


# ---------------------------------------------------------------------------
# run / sweep / paired


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    for w in cfg.validate():
        print(f"warning: {w}", file=sys.stderr)
    run_dir, summary = run_sync(cfg, args.split, Path(args.runs_dir), label=args.label)
    _print_summary(run_dir, summary)
    return 0


def _print_summary(run_dir: Path, s: dict) -> None:
    o = s["outcomes"]
    cost = s["cost"]["window"]["cost"]
    print(f"{run_dir}")
    print(f"  window {s['window_s']:.1f}s | cost {'unavailable' if cost is None else f'{cost:.4f}'} | "
          f"success {o['success']} / valid {s['valid_episodes']} / submitted {s['cohort_size']} | "
          f"infra {o['infra_error']} | wall timeouts {s['wall_timeouts']}")
    for w in s["warnings"]:
        print(f"  ! {w}")


def _check_overrides(overrides: dict) -> None:
    bad = sorted(set(flatten(overrides)) - TUNABLE_KEYS)
    if bad:
        raise ConfigError(f"sweep variants may only change tunable keys {sorted(TUNABLE_KEYS)}; got {bad}")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cmd_sweep(args) -> int:
    path = Path(args.config)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent / spec["base"]
    variants = spec["variants"]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ConfigError("variant names must be unique")
    baseline_name = spec["baseline_variant"]
    if baseline_name not in names:
        raise ConfigError(f"baseline_variant {baseline_name!r} is not a variant")
    if len(variants) - 1 > MAX_SWEEP_CANDIDATES:
        raise ConfigError(f"at most {MAX_SWEEP_CANDIDATES} candidate configurations per sweep")
    for v in variants:
        _check_overrides(v.get("overrides") or {})
    split = args.split or spec.get("split", "dev")
    out_dir = Path(args.experiments_dir) / f"{spec['name']}-{_stamp()}"
    out_dir.mkdir(parents=True)

    runs: dict[str, Path] = {}
    for v in variants:
        cfg = load_config(base, v.get("overrides") or {})
        cfg.name = f"{spec['name']}-{v['name']}"
        print(f"== variant {v['name']}")
        run_dir, summary = run_sync(cfg, split, Path(args.runs_dir), label=f"sweep:{spec['name']}:{v['name']}")
        _print_summary(run_dir, summary)
        runs[v["name"]] = run_dir

    # Development selection has one trial per variant, so the trial-count and
    # spread checks cannot apply; they are relaxed here and the result is
    # labelled as needing paired validation.
    rule = {"min_trials": 1, "require_reduction_above_spread": False, **(spec.get("acceptance") or {})}
    baseline = load_run(runs[baseline_name])
    results = {}
    for name, run_dir in runs.items():
        if name == baseline_name:
            continue
        comp = compare_runs(baseline, load_run(run_dir))
        results[name] = {"run": str(run_dir), "comparison": comp, "acceptance": evaluate_acceptance([comp], rule)}
    accepted = {n: r for n, r in results.items() if r["acceptance"]["accepted"]}
    chosen = min(accepted, key=lambda n: results[n]["comparison"]["candidate_summary"]["window_s"]) if accepted else baseline_name
    chosen_cfg = load_run(runs[chosen]).manifest["config"]
    (out_dir / "chosen.yaml").write_text(yaml.safe_dump(chosen_cfg, sort_keys=False), encoding="utf-8")
    selection = {"sweep": spec["name"], "split": split, "baseline_variant": baseline_name, "chosen": chosen,
                 "runs": {n: str(p) for n, p in runs.items()},
                 "reason": "lowest allocated window among accepted candidates" if accepted else "no candidate met the acceptance rule; baseline kept",
                 "note": "Development selection from one trial per variant; validate on held-out tasks with paired trials."}
    (out_dir / "selection.json").write_text(json.dumps({**selection, "results": results}, indent=2, default=str), encoding="utf-8")
    lines = [f"# Sweep {spec['name']} ({split})", "", f"Chosen: **{chosen}** ({selection['reason']}).", "", selection["note"], ""]
    for name, r in results.items():
        lines += [f"## {name} vs {baseline_name}", "", render_comparison(r["comparison"], r["acceptance"]).split("\n", 1)[1]]
    (out_dir / "selection.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"chosen variant: {chosen} -> {out_dir / 'chosen.yaml'}")
    return 0


def cmd_paired(args) -> int:
    base_cfg = load_config(args.baseline)
    cand_cfg = load_config(args.candidate)
    name = args.name or f"paired-{_stamp()}"
    pairs = []
    for trial in range(args.trials):
        # Alternate order so warm-host and time-of-day effects do not favor one side.
        order = [("baseline", base_cfg), ("candidate", cand_cfg)]
        if trial % 2 == 1:
            order.reverse()
        produced = {}
        for role, cfg in order:
            print(f"== trial {trial + 1}/{args.trials}: {role}")
            run_dir, summary = run_sync(cfg, args.split, Path(args.runs_dir), label=f"paired:{name}:t{trial}:{role}")
            _print_summary(run_dir, summary)
            produced[role] = run_dir.as_posix()
        pairs.append({"baseline": produced["baseline"], "candidate": produced["candidate"],
                      "order": [role for role, _ in order]})
    rule = yaml.safe_load(Path(args.acceptance).read_text(encoding="utf-8")) if args.acceptance else {}
    experiment = {"name": name, "split": args.split, "status": args.status, "pairs": pairs,
                  "acceptance": {**DEFAULT_ACCEPTANCE, **(rule or {})}}
    exp_path = Path(args.experiments_dir) / f"{name}.yaml"
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    print(f"experiment written to {exp_path}")
    return _report_experiment(exp_path, Path(args.reports_dir))


# ---------------------------------------------------------------------------
# compare / report


def cmd_compare(args) -> int:
    comp = compare_runs(load_run(args.baseline), load_run(args.candidate))
    acceptance = evaluate_acceptance([comp], yaml.safe_load(Path(args.acceptance).read_text()) if args.acceptance else None)
    text = render_comparison(comp, acceptance)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"written to {args.out}")
    else:
        print(text)
    return 0


def _report_experiment(path: Path, reports_dir: Path) -> int:
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    while not (root / "pyproject.toml").exists() and root.parent != root:
        root = root.parent
    comparisons = []
    synthetic = False
    for pair in exp["pairs"]:
        base, cand = load_run(root / pair["baseline"]), load_run(root / pair["candidate"])
        synthetic |= base.manifest["synthetic"] or cand.manifest["synthetic"]
        comparisons.append(compare_runs(base, cand))
    acceptance = evaluate_acceptance(comparisons, exp.get("acceptance"))
    text = render_experiment(exp["name"], comparisons, acceptance, {"status": exp.get("status"), "synthetic": synthetic})
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{exp['name']}.md"
    out.write_text(text, encoding="utf-8")
    verdict = "A/A CALIBRATION" if acceptance["aa_calibration"] else ("ACCEPTED" if acceptance["accepted"] else "NOT ACCEPTED")
    print(f"{verdict}: report written to {out}")
    return 0


def cmd_report(args) -> int:
    if args.run:
        run = load_run(args.run)
        text = render_run(run.manifest, run.summary)
        (run.path / "report.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    return _report_experiment(Path(args.experiment), Path(args.reports_dir))


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rollout-cost", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-fixtures", help="check starting code fails and reference fixes pass")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="dev")
    p.set_defaults(func=cmd_validate_fixtures)

    p = sub.add_parser("doctor", help="check configuration, fixtures, executor, GPU and model server")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="dev")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run", help="run one cohort under one configuration")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--label")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="run a bounded development sweep and select a candidate")
    p.add_argument("--config", required=True)
    p.add_argument("--split")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--experiments-dir", default="experiments")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("paired", help="run alternating baseline/candidate trials and evaluate acceptance")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--split", default="heldout")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--name")
    p.add_argument("--status", default="preliminary")
    p.add_argument("--acceptance", help="YAML file with acceptance-rule overrides")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--reports-dir", default="reports")
    p.set_defaults(func=cmd_paired)

    p = sub.add_parser("compare", help="compare two runs")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--acceptance")
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report", help="render a run or experiment report")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--experiment")
    p.add_argument("--reports-dir", default="reports")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
