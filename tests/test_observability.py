"""Compatibility between the runner, exporter, export schema, mock server, dashboards and deployments.

Postgres integration tests run when RC_TEST_PG is set to a database URL, e.g. the
compose database: RC_TEST_PG=postgresql://rollout:rollout@127.0.0.1:55432/rollout
"""

import asyncio
import importlib.util
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from rollout_cost.analysis import evaluate_sweep_selection, load_run
from rollout_cost.config import load_config
from rollout_cost.contract import DATASOURCE_UIDS, LIVE_METRICS, STATE_CODES, VLLM_DASHBOARD_METRICS
from rollout_cost.export import experiment_rows, run_rows, schema_tables
from rollout_cost.pinning import split_cpus, write_pinned
from rollout_cost.runner import execute_run
from rollout_cost.spans import build_spans, concurrency, totals_by_kind
from rollout_cost.telemetry import SERVER_SIGNALS, parse_prometheus

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "observability"
# Exported into the test database and removed afterwards; never a real run id.
TEST_RUN_ID = "pytest-observability"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_dashboards", OBS / "build_dashboards.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dashboards() -> dict[str, dict]:
    return {p.relative_to(OBS / "grafana" / "dashboards").as_posix(): json.loads(p.read_text())
            for p in (OBS / "grafana" / "dashboards").rglob("*.json")}


def panel_targets(dash: dict):
    for panel in dash["panels"]:
        for target in panel.get("targets", []):
            yield panel, target


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "synthetic.yaml", {
        "model": {"scripted_base_latency_s": 0.02}, "workload": {"repetitions": 1}})
    run_dir, _ = asyncio.run(execute_run(cfg, "dev", tmp_path_factory.mktemp("runs"), run_id=TEST_RUN_ID))
    return load_run(run_dir)


# -- dashboards ---------------------------------------------------------------


def test_committed_dashboards_match_generator():
    rendered = load_generator().render()
    assert set(rendered) == set(dashboards())
    for path, content in rendered.items():
        committed = (OBS / "grafana" / "dashboards" / path).read_text(encoding="utf-8")
        assert committed == content, f"{path} is stale: run `uv run python observability/build_dashboards.py`"


@pytest.mark.parametrize("path,title,field,value", [
    ("history/run-detail.json", "Data", "data", "Measured run"),
    ("history/run-detail.json", "Success / valid / submitted", "outcome", "2 / 6 / 6"),
    ("history/experiments.json", "Decision", "verdict", "A/A CALIBRATION"),
    ("history/experiments.json", "Data", "data", "measured"),
    ("history/experiments.json", "Kind / status", "kind", "paired"),
])
def test_stat_verification_detects_hidden_text_values(path, title, field, value):
    """A successful SQL response must also select a field the stat can display."""
    spec = importlib.util.spec_from_file_location("verify_dashboards", ROOT / "scripts/verify_dashboards.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    p = next(p for p in dashboards()[path]["panels"] if p["title"] == title)
    response = {"results": {"A": {"frames": [{"schema": {"fields": [{"name": field, "type": "string"}]},
                                             "data": {"values": [[value]]}}]}}}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
                      base_url="http://grafana.test") as http:
        assert verifier.query(http, p["targets"][0], {}, "now-1d", "now", panel=p) == ("ok", 1)
        broken = json.loads(json.dumps(p))
        broken["options"]["reduceOptions"]["fields"] = ""
        status, rows = verifier.query(http, p["targets"][0], {}, "now-1d", "now", panel=broken)
        assert status.startswith("ERROR stat panel has text data") and rows == 1


def test_promql_uses_only_contract_and_known_vllm_metrics():
    known = set(LIVE_METRICS) | set(VLLM_DASHBOARD_METRICS)
    for dash in dashboards().values():
        for _, target in panel_targets(dash):
            for name in re.findall(r"\b(rc_[a-z_]+|vllm:[a-z_:]+)", target.get("expr", "")):
                assert name in known, f"{dash['uid']}: unknown metric {name}"
    variables = [v for d in dashboards().values() for v in d["templating"]["list"] if v["datasource"]["type"] == "prometheus"]
    for v in variables:
        for name in re.findall(r"\b(rc_[a-z_]+)", v["definition"]):
            assert name in LIVE_METRICS


def test_sql_references_existing_tables():
    tables = schema_tables()
    for dash in dashboards().values():
        queries = [t["rawSql"] for _, t in panel_targets(dash) if "rawSql" in t]
        queries += [v["query"] for v in dash["templating"]["list"] if v["datasource"]["type"] != "prometheus"]
        for query in queries:
            for table in re.findall(r"\b(?:FROM|JOIN)\s+([a-z_]+)", query):
                assert table in tables, f"{dash['uid']}: unknown table {table} in {query[:80]}"


def test_datasource_uids_and_types_match_provisioning():
    provisioned = {}
    for path in (OBS / "grafana" / "provisioning" / "datasources").glob("*.yaml"):
        for ds in yaml.safe_load(path.read_text())["datasources"]:
            provisioned[ds["uid"]] = ds["type"]
    assert set(provisioned) == set(DATASOURCE_UIDS.values())
    for dash in dashboards().values():
        refs = [p["datasource"] for p in dash["panels"] if p.get("datasource")]
        refs += [t["datasource"] for _, t in panel_targets(dash)] + [v["datasource"] for v in dash["templating"]["list"]]
        for ref in refs:
            assert provisioned.get(ref["uid"]) == ref["type"], f"{dash['uid']}: {ref}"


def test_timeline_maps_every_state_code():
    detail = dashboards()["history/run-detail.json"]
    timeline = next(p for p in detail["panels"] if p["type"] == "state-timeline")
    mapped = timeline["fieldConfig"]["defaults"]["mappings"][0]["options"]
    assert {int(k) for k in mapped} == set(STATE_CODES.values())


def test_compose_defaults_match_versions_file():
    versions = dict(line.split("=", 1) for line in (OBS / "versions.env").read_text().splitlines()
                    if line and not line.startswith("#"))
    compose = (OBS / "compose.yaml").read_text()
    for key, value in versions.items():
        assert f"${{{key}:-{value}}}" in compose, f"compose default for {key} differs from versions.env"


def test_provisioning_paths_are_environment_driven():
    for path in (OBS / "grafana" / "provisioning").rglob("*.yaml"):
        text = path.read_text()
        assert "/etc/rollout-cost" not in text and "127.0.0.1" not in text, f"{path.name} hard-codes a deployment"


# -- mock server ↔ telemetry ↔ dashboards ----------------------------------------


def test_mock_metrics_cover_signals_and_dashboard_metrics():
    from rollout_cost.mock_server import MockVLLM

    metrics = parse_prometheus(MockVLLM({}, {}).metrics_text())
    for signal, names in SERVER_SIGNALS.items():
        assert any(n in metrics for n in names), f"mock does not expose {signal}"
    exposed = {name.split("{")[0] for name in metrics}
    missing = {m for m in VLLM_DASHBOARD_METRICS if m not in exposed and m != "vllm:gpu_cache_usage_perc"}
    assert not missing, missing


def test_mock_vllm_run_through_restart_and_drain(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = load_config(ROOT / "configs" / "mock-vllm.yaml", {
        "model": {"base_url": f"http://127.0.0.1:{port}/v1"},
        "server": {"args": {"port": port, "base-latency": 0.02}},
        "workload": {"repetitions": 1}, "telemetry": {"live_metrics_port": None}})
    run_dir, summary = asyncio.run(execute_run(cfg, "dev", tmp_path, run_id="mockrun"))
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["synthetic"] and "mock" in manifest["synthetic_reason"]
    assert manifest["timeline"]["server_prep"]["reset"]["method"] == "restart"
    assert summary["drain"]["server_idle"]["verified"] is True
    assert summary["outcomes"]["infra_error"] == 0 and summary["successes"] == 4  # mixed policy: 4 of 6 tasks
    assert summary["tokens"]["usage_complete"] and summary["prefix_cache"]["available"]
    assert (run_dir / "server.log").exists()


# -- spans and export rows ---------------------------------------------------------


def test_spans_match_episode_accounting(synthetic_run):
    events = [json.loads(line) for line in (synthetic_run.path / "events.jsonl").read_text().splitlines()]
    totals = totals_by_kind(build_spans(events))
    fields = {"llm_wait": "llm_wait_s", "model_request": "request_s", "tool_wait": "tool_wait_s", "tool": "tool_s",
              "file_op": "file_op_s"}
    for e in synthetic_run.episodes:
        t = totals[e["episode_id"]]
        for kind, field in fields.items():
            assert abs(t.get(kind, 0.0) - e[field]) <= 0.05 + 0.02 * e[field], (e["episode_id"], kind)
        grading = t.get("grading", 0.0) + t.get("grading_wait", 0.0)
        assert abs(grading - e["grading_s"]) <= 0.1 + 0.05 * e["grading_s"]


def test_concurrency_series_respect_caps(synthetic_run):
    events = [json.loads(line) for line in (synthetic_run.path / "events.jsonl").read_text().splitlines()]
    series = concurrency(build_spans(events))
    s = synthetic_run.summary
    assert max(v for _, v in series["active_episodes"]) <= s["max_active_episodes"]
    assert max(v for _, v in series["llm_inflight"]) <= s["max_inflight_llm_requests"]
    assert max(v for _, v in series["tool_busy"]) <= s["tool_workers"]
    assert all(v >= 0 for points in series.values() for _, v in points)
    assert all(points[-1][1] == 0 for points in series.values() if points)


def test_export_rows_match_schema_columns(synthetic_run):
    tables = schema_tables()
    rows = run_rows(synthetic_run)
    for table, items in rows.items():
        for item in items:
            assert set(item) == set(tables[table]), table
    assert len(rows["episodes"]) == synthetic_run.summary["cohort_size"]
    assert rows["runs"][0]["window_cost"] == synthetic_run.summary["cost"]["window"]["cost"]


# -- portability ----------------------------------------------------------------------


def test_run_artifacts_are_portable(synthetic_run, tmp_path):
    for name in ("run.json", "summary.json"):
        text = (synthetic_run.path / name).read_text()
        assert "\\\\" not in text, f"{name} contains a Windows path"
    moved = tmp_path / "copied" / synthetic_run.path.name
    shutil.copytree(synthetic_run.path, moved)
    reloaded = load_run(str(moved).replace("/", "\\") if os.name == "nt" else moved)
    assert reloaded.summary == synthetic_run.summary


def test_sweep_selection_with_relative_posix_paths(synthetic_run, tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    runs = tmp_path / "runs"
    shutil.copytree(synthetic_run.path, runs / "base")
    shutil.copytree(synthetic_run.path, runs / "cand")
    sweep = tmp_path / "experiments" / "s1"
    sweep.mkdir(parents=True)
    (sweep / "selection.json").write_text(json.dumps({"sweep": "s", "split": "dev", "baseline_variant": "a", "chosen": "a",
                                                      "runs": {"a": "runs/base", "b": "runs\\cand"}}))
    monkeypatch.chdir(tmp_path)
    ev = evaluate_sweep_selection(sweep)
    rows = experiment_rows(ev)
    assert rows["experiments"][0]["verdict"] == "CHOSEN: a"
    assert len(rows["experiment_trials"]) == 2


# -- pinning (Lightning) ----------------------------------------------------------------


@pytest.mark.parametrize("cpus, reserved, workers", [
    ([0, 1], [], []),
    ([0, 1, 2, 3], [0, 1], [2, 3]),
    (list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7]),
    (list(range(16)), list(range(8)), [8, 9, 10, 11]),
])
def test_split_cpus(cpus, reserved, workers):
    assert split_cpus(cpus) == (reserved, workers)


@pytest.mark.parametrize("rehearsal", [True, False])
def test_pinned_configs_load_and_have_no_placeholders(tmp_path, rehearsal):
    (tmp_path / "pyproject.toml").write_text("")
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    write_pinned(configs, revision="abc123", vllm_bin="/opt/vllm/bin/vllm", cpus=list(range(8)), rate=0.8,
                 rate_source="test", executor="local", rehearsal=rehearsal)
    for name in ("lightning.pinned.yaml", "smoke.pinned.yaml"):
        cfg = load_config(configs / name)
        flat = json.dumps(cfg.to_dict())
        assert "REPLACE" not in flat
        assert cfg.tools.worker_cpus == [4, 5, 6, 7] and cfg.server.reserved_cpus == [0, 1, 2, 3]
        assert cfg.server.start_command[:3] == ["taskset", "-c", "0,1,2,3"]
        if rehearsal:
            from rollout_cost.config import render_command
            from rollout_cost.mock_server import build_parser

            command = render_command(cfg.server.start_command, cfg.server.args)
            build_parser().parse_args(command[command.index("rollout_cost.mock_server") + 1:])  # mock accepts every flag
    smoke = load_config(configs / "smoke.pinned.yaml")
    assert smoke.execution.max_active_episodes == 1 and smoke.workload.repetitions == 1
    sweep = yaml.safe_load((configs / "sweep.pinned.yaml").read_text())
    assert sweep["base"] == "lightning.pinned.yaml"


# -- Linux-only behavior --------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU affinity is Linux-only")
def test_local_executor_pins_to_slot_cpu(tmp_path):
    from rollout_cost.tools import LocalExecutor, ToolSlot

    cpu = sorted(os.sched_getaffinity(0))[-1]
    result = asyncio.run(LocalExecutor().run(tmp_path, ["python", "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
                                             10, ToolSlot(0, cpu), "affinity"))
    assert result.output.strip() == f"[{cpu}]"


# -- PostgreSQL integration (RC_TEST_PG) ---------------------------------------------------

PG_URL = os.environ.get("RC_TEST_PG")
needs_pg = pytest.mark.skipif(not PG_URL, reason="set RC_TEST_PG to a PostgreSQL URL")


@pytest.fixture
def exported(synthetic_run, tmp_path):
    """Export the fixture run, and delete its rows afterwards so a shared database stays clean."""
    import psycopg

    from rollout_cost.export import RUN_TABLES, export

    runs = tmp_path / "runs"
    shutil.copytree(synthetic_run.path, runs / synthetic_run.path.name)
    first = export(PG_URL, runs, tmp_path / "no-experiments")
    yield runs, first
    with psycopg.connect(PG_URL) as conn:
        for table in RUN_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE run_id = %s", (TEST_RUN_ID,))


@needs_pg
def test_export_is_idempotent(exported, tmp_path):
    import psycopg

    from rollout_cost.export import export

    runs, first = exported
    count = lambda conn, t: conn.execute(f"SELECT count(*) FROM {t} WHERE run_id = %s", (TEST_RUN_ID,)).fetchone()[0]  # noqa: E731
    tables = ("runs", "episodes", "spans", "episode_states", "concurrency", "requests")
    with psycopg.connect(PG_URL) as conn:
        before = {t: count(conn, t) for t in tables}
    second = export(PG_URL, runs, tmp_path / "no-experiments")
    with psycopg.connect(PG_URL) as conn:
        after = {t: count(conn, t) for t in tables}
    assert first["rows"] == second["rows"] and before == after and before["runs"] == 1


@needs_pg
def test_every_dashboard_query_executes(exported):
    import psycopg

    with psycopg.connect(PG_URL) as conn:
        experiment = (conn.execute("SELECT name FROM experiments ORDER BY name LIMIT 1").fetchone() or ["none"])[0]
        for dash in dashboards().values():
            for panel, target in panel_targets(dash):
                if "rawSql" not in target:
                    continue
                query = target["rawSql"].replace("$run", TEST_RUN_ID).replace("$experiment", experiment)
                rows = conn.execute(query).fetchall()  # every query must at least execute
                if f"run_id = '{TEST_RUN_ID}'" in query and panel["title"] not in (
                        "GPU utilization", "GPU memory used", "Server requests and KV cache", "Warnings"):
                    assert rows, f"{dash['uid']}/{panel['title']} returned no rows"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
