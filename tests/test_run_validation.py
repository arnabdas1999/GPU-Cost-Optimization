"""Decision guards for incomplete artifacts and unverified real-server idleness."""

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from rollout_cost.analysis import RunFormatError, compare_runs, evaluate_acceptance, load_run, summarize_run
from rollout_cost.cli import main
from rollout_cost.config import load_config
from rollout_cost.runner import execute_run

ROOT = Path(__file__).resolve().parents[1]
# Isolate validity checks from the performance thresholds in these tests.
PERMISSIVE_TIMING = {"min_trials": 1, "min_window_reduction_pct": -1000,
                     "require_reduction_every_pair": False, "require_reduction_above_spread": False}


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "synthetic.yaml", {
        "model": {"scripted_policy": "noop", "scripted_base_latency_s": 0.0,
                  "scripted_per_token_latency_s": 0.0}, "workload": {"repetitions": 1}})
    path, _ = asyncio.run(execute_run(cfg, "dev", tmp_path_factory.mktemp("validity"), run_id="complete"))
    return load_run(path)


def refresh(run):
    run.summary = summarize_run(run.manifest, run.episodes, run.manifest["timeline"])


def pair(source, measured=False):
    base, cand = deepcopy(source), deepcopy(source)
    cand.manifest["run_id"] = "candidate"
    cand.manifest["config"]["execution"]["max_active_episodes"] += 1
    for run in (base, cand):
        if measured:
            # Exercise measured-run validation without needing a GPU server.
            run.manifest["synthetic"] = False
            run.manifest["timeline"]["server_prep"] = {"idle_before_run": {"verified": True}}
            run.manifest["timeline"]["drain"]["server_idle"] = {"verified": True}
        refresh(run)
    return base, cand


@pytest.mark.parametrize("measured", [False, True])
def test_complete_consistent_runs_can_be_accepted(completed_run, measured):
    comp = compare_runs(*pair(completed_run, measured))
    assert comp["valid"], comp["problems"]
    assert evaluate_acceptance([comp], PERMISSIVE_TIMING)["accepted"]


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("missing", [False, True])
def test_unverified_idleness_blocks_acceptance(completed_run, phase, missing):
    base, cand = pair(completed_run, measured=True)
    timeline = cand.manifest["timeline"]
    parent, key = ((timeline["server_prep"], "idle_before_run") if phase == "before"
                   else (timeline["drain"], "server_idle"))
    if missing:
        parent.pop(key)
    else:
        parent[key] = {"verified": False, "reason": "server still busy"}
    refresh(cand)
    comp = compare_runs(base, cand)
    assert not comp["valid"]
    assert any(f"idleness {phase} the cohort was not verified" in p for p in comp["problems"])
    assert any(f"idleness {phase} the cohort was not verified" in w for w in cand.summary["warnings"])
    assert not evaluate_acceptance([comp], PERMISSIVE_TIMING)["accepted"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unexpected"])
def test_matching_but_damaged_episode_files_are_rejected(completed_run, damage):
    base, cand = pair(completed_run)
    for run in (base, cand):
        if damage == "missing":
            run.episodes.pop()
        elif damage == "duplicate":
            run.episodes.append(deepcopy(run.episodes[0]))
        else:
            run.episodes[-1]["episode_id"] = "unexpected-episode"
        # Even a summary rebuilt from the damaged file cannot validate it.
        refresh(run)
    comp = compare_runs(base, cand)
    assert not comp["valid"]
    assert any("exactly one terminal record" in p for p in comp["problems"])
    assert not evaluate_acceptance([comp], PERMISSIVE_TIMING)["accepted"]


@pytest.mark.parametrize("field", ["successes", "cohort_size", "valid_episodes", "window_s",
                                   "cost", "service_p90_s", "outcomes", "run_id", "synthetic"])
def test_stale_or_inconsistent_summary_is_rejected(completed_run, field):
    base, cand = pair(completed_run)
    if field == "cost":
        cand.summary["cost"]["window"]["cost"] = -1
    elif field == "service_p90_s":
        cand.summary["latency"]["service_s"]["p90"] = -1
    elif field == "outcomes":
        cand.summary["outcomes"]["infra_error"] += 1
    elif field == "run_id":
        cand.summary["run_id"] = "different-run"
    elif field == "synthetic":
        cand.summary["synthetic"] = not cand.summary["synthetic"]
    else:
        cand.summary[field] += 1
    comp = compare_runs(base, cand)
    assert not comp["valid"]
    assert any("summary disagrees" in p and field in p for p in comp["problems"])
    assert not evaluate_acceptance([comp], PERMISSIVE_TIMING)["accepted"]


def test_unfinished_run_with_a_summary_is_rejected(completed_run):
    base, cand = pair(completed_run)
    cand.manifest["status"] = "aborted"
    comp = compare_runs(base, cand)
    assert not comp["valid"] and any("run status" in p for p in comp["problems"])
    assert not evaluate_acceptance([comp], PERMISSIVE_TIMING)["accepted"]


def test_missing_summary_has_an_actionable_error(completed_run):
    base, cand = pair(completed_run)
    cand.summary = {}
    with pytest.raises(RunFormatError, match="incomplete or invalid run artifacts"):
        compare_runs(base, cand)


def test_cli_reports_missing_summary_without_a_traceback(completed_run, tmp_path, capsys):
    (tmp_path / "run.json").write_text((completed_run.path / "run.json").read_text(), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text((completed_run.path / "episodes.jsonl").read_text(), encoding="utf-8")
    assert main(["compare", "--baseline", str(completed_run.path), "--candidate", str(tmp_path)]) == 2
    assert "run artifact error" in capsys.readouterr().err
