"""Reference valuations preserve measurements and stay explicit in reports/exports."""

import asyncio
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rollout_cost.analysis import (Evaluated, compare_runs, evaluate_acceptance, load_run, price_run, run_problems)
from rollout_cost.cli import main
from rollout_cost.config import ConfigError, load_config
from rollout_cost.cost import ESTIMATED_COST_PREFIX
from rollout_cost.export import experiment_rows, run_rows
from rollout_cost.pinning import write_pinned
from rollout_cost.runner import execute_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    cfg = load_config(ROOT / "configs/synthetic.yaml", {
        "model": {"scripted_policy": "noop", "scripted_base_latency_s": 0, "scripted_per_token_latency_s": 0},
        "workload": {"repetitions": 1}, "cost": {"allocation_hourly_rate": None}})
    path, _ = asyncio.run(execute_run(cfg, "dev", tmp_path_factory.mktemp("cost-runs"), run_id="cost-test"))
    return load_run(path)


def reference():
    return asdict(load_config(ROOT / "configs/notebook.yaml").cost)


def test_existing_run_valuation_keeps_artifacts_and_original_validity(measured):
    names = ("run.json", "summary.json", "episodes.jsonl", "report.md")
    original_files = {name: (measured.path / name).read_bytes() for name in names}
    original_summary = deepcopy(measured.summary)
    priced = price_run(measured, reference())
    assert priced.summary["cost"]["window"]["cost"] == pytest.approx(measured.summary["window_s"] / 3600 * 0.539999)
    assert priced.summary["cost"]["run_inclusive"]["cost"] == pytest.approx(measured.summary["run_inclusive_s"] / 3600 * 0.539999)
    assert priced.summary["cost"]["per_success"] is None  # no successful tasks
    assert priced.summary["outcomes"] == measured.summary["outcomes"]
    assert priced.summary["window_s"] == measured.summary["window_s"]
    assert any(w.startswith(ESTIMATED_COST_PREFIX) for w in priced.summary["warnings"])
    assert not any("Allocation price unknown" in w for w in priced.summary["warnings"])
    assert measured.summary == original_summary and priced.manifest is measured.manifest
    assert run_problems(measured) == []
    assert {name: (measured.path / name).read_bytes() for name in names} == original_files


def test_report_cli_writes_separate_valuation(measured, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(measured.path, run_dir)
    original = {name: (run_dir / name).read_bytes() for name in ("run.json", "summary.json", "report.md")}
    assert main(["report", "--run", str(run_dir), "--cost-config", str(ROOT / "configs/notebook.yaml")]) == 0
    text = (run_dir / "report.estimated.md").read_text()
    assert "Estimated equivalent cost" in text and "cloud.google.com" in text
    assert "Post-run valuation" in text and "0.54 USD/hour" in text
    assert {name: (run_dir / name).read_bytes() for name in original} == original


def test_report_works_through_a_legacy_encoded_pipe(measured, tmp_path):
    run_dir = tmp_path / "piped-run"
    shutil.copytree(measured.path, run_dir)
    result = subprocess.run([sys.executable, "-m", "rollout_cost.cli", "report", "--run", str(run_dir),
                             "--cost-config", str(ROOT / "configs/notebook.yaml")], capture_output=True,
                            env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "Estimated equivalent cost" in result.stdout.decode("utf-8")


def test_export_values_runs_and_experiment_trials_consistently(measured):
    comp = compare_runs(measured, measured)
    acceptance = evaluate_acceptance([comp])
    ev = Evaluated("cost-aa", "paired", {"split": "dev"}, [measured], {"candidate": [measured]},
                   {"candidate": [comp]}, {"candidate": acceptance}, synthetic=True)
    before = deepcopy(ev.acceptance)
    trials = experiment_rows(ev, reference())["experiment_trials"]
    rows = run_rows(price_run(measured, reference()))
    amount = rows["runs"][0]["window_cost"]
    assert amount > 0 and all(t["window_cost"] == amount for t in trials)
    assert ev.acceptance == before and not ev.acceptance["candidate"]["accepted"]
    assert measured.summary["cost"]["window"]["cost"] is None
    assert any(r["message"].startswith(ESTIMATED_COST_PREFIX) for r in rows["run_warnings"])


def test_explicit_pin_rate_does_not_inherit_reference_source(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    write_pinned(configs, revision="fixed", vllm_bin="vllm", cpus=[0, 1], profile="notebook",
                 rate=2.0, rate_source="Actual provider allocation")
    cost = load_config(configs / "notebook.pinned.yaml").cost
    assert cost.allocation_hourly_rate == 2 and cost.rate_kind == "recorded"
    assert cost.rate_source == "Actual provider allocation"


@pytest.mark.parametrize("cost", [
    {"allocation_hourly_rate": float("nan")}, {"allocation_hourly_rate": float("inf")},
    {"allocation_hourly_rate": -1}, {"rate_kind": "estimated_equivalent", "allocation_hourly_rate": None},
    {"rate_kind": "estimated_equivalent", "allocation_hourly_rate": 1, "rate_source": None},
])
def test_invalid_or_unattributed_estimates_are_rejected(cost):
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs/synthetic.yaml", {"cost": cost})


@pytest.mark.skipif(not os.environ.get("RC_TEST_PG"), reason="set RC_TEST_PG to test PostgreSQL valuations")
def test_database_valuation_is_idempotent_and_reversible(measured, tmp_path):
    import psycopg
    from rollout_cost.export import RUN_TABLES, export

    url = os.environ["RC_TEST_PG"]
    runs = tmp_path / "runs"
    shutil.copytree(measured.path, runs / measured.path.name)
    rid = measured.manifest["run_id"]
    before = (measured.path / "summary.json").read_bytes()
    try:
        for _ in range(2):
            result = export(url, runs, tmp_path / "empty", cost_config=reference())
            assert result["runs"] == 1 and not result["skipped"]
        with psycopg.connect(url) as conn:
            prices = conn.execute("SELECT window_cost FROM runs WHERE run_id = %s", (rid,)).fetchall()
            assert len(prices) == 1 and prices[0][0] > 0
            warnings = conn.execute("SELECT message FROM run_warnings WHERE run_id = %s", (rid,)).fetchall()
            assert any(w[0].startswith(ESTIMATED_COST_PREFIX) for w in warnings)
        export(url, runs, tmp_path / "empty")
        with psycopg.connect(url) as conn:
            assert conn.execute("SELECT window_cost FROM runs WHERE run_id = %s", (rid,)).fetchone()[0] is None
        assert (measured.path / "summary.json").read_bytes() == before
    finally:
        with psycopg.connect(url) as conn:
            for table in RUN_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE run_id = %s", (rid,))
