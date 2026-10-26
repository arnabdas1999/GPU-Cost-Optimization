"""Notebook profile, artifact handling and real HTTP rehearsal without a GPU."""

import ast
import asyncio
from copy import deepcopy
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile

import pytest

from rollout_cost.cli import main
from rollout_cost.config import load_config
from rollout_cost.notebook import archive_results, check_gpu, check_result, latest_run, run_logged
from rollout_cost.runner import execute_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cpus", ["0,1", "2,4,6,8"])
def test_notebook_pin_preserves_model_context_and_cpu_contract(tmp_path, cpus):
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    assert main(["pin", "--profile", "notebook", "--configs-dir", str(configs),
                 "--revision", "exact-sha", "--vllm", "/tmp/vllm/bin/vllm", "--cpus", cpus]) == 0
    for name in ("notebook.pinned.yaml", "notebook-smoke.pinned.yaml"):
        cfg = load_config(configs / name)
        assert cfg.model.repository == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
        assert cfg.model.revision == cfg.model.tokenizer_revision == "exact-sha"
        command = cfg.server.start_command
        assert command[command.index("serve") + 1] == cfg.model.repository
        assert cfg.server.args["revision"] == cfg.server.args["tokenizer-revision"] == "exact-sha"
        assert cfg.server.args["dtype"] == "float16"
        assert cfg.server.args["max-model-len"] == cfg.workload.context_window_tokens == 8192
        assert cfg.workload.max_steps == 12
        assert cfg.workload.edit_mode == "whole_file"
        assert cfg.generation.max_output_tokens_per_call == 1024
        assert cfg.generation.temperature == 0 and cfg.generation.top_p == 1
        assert cfg.server.args["tensor-parallel-size"] == 1
        assert cfg.tools.executor == "local" and cfg.telemetry.live_metrics_port is None
        assert set(cfg.tools.worker_cpus + cfg.server.reserved_cpus) <= {int(c) for c in cpus.split(",")}
        assert not set(cfg.tools.worker_cpus) & set(cfg.server.reserved_cpus)
        assert cfg.cost.allocation_hourly_rate is None
    smoke = load_config(configs / "notebook-smoke.pinned.yaml")
    assert smoke.workload.repetitions == smoke.execution.max_active_episodes == smoke.execution.max_inflight_llm_requests == 1


@pytest.mark.parametrize("capability,works", [("7.5", True), ("6.0", False)])
def test_gpu_check_rejects_p100_before_downloads(tmp_path, monkeypatch, capability, works):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, f"Test GPU, 15360, {capability}, 570.0\n"))
    if works:
        check_gpu(tmp_path)
    else:
        with pytest.raises(RuntimeError, match="P100 is unsuitable"):
            check_gpu(tmp_path)
    assert (tmp_path / "notebook-hardware.txt").exists()


def test_logged_command_records_output_and_nonzero_exit(tmp_path):
    run_logged(tmp_path, [sys.executable, "-c", "print('visible output')"], "notebook-test.log")
    assert "visible output" in (tmp_path / "notebook-test.log").read_text()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        run_logged(tmp_path, [sys.executable, "-c", "print('failure detail'); raise SystemExit(3)"], "notebook-test.log")
    assert "failure detail" in (tmp_path / "notebook-test.log").read_text()


def test_archive_includes_results_but_excludes_packages_models_and_env(tmp_path):
    for name in ("runs/r/run.json", "configs/notebook.pinned.yaml", ".venv/secret", ".env",
                 "model-cache/weights", "notebook-setup.log"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    with tarfile.open(archive_results(tmp_path)) as archive:
        files = archive.getnames()
    assert "runs/r/run.json" in files and "configs/notebook.pinned.yaml" in files
    assert "notebook-setup.log" in files
    assert not any(name.startswith((".venv", ".env", "model-cache")) for name in files)


def test_notebook_cells_are_valid_python_and_have_no_saved_outputs():
    notebook = json.loads((ROOT / "notebooks/gpu_smoke_test.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_enabling_pilot_keeps_the_existing_smoke_label():
    notebook = json.loads((ROOT / "notebooks/gpu_smoke_test.ipynb").read_text())
    settings = "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == "settings"))
    namespace = {"datetime": datetime, "timezone": timezone, "SESSION_TAG": "existing-smoke-session"}
    exec(settings, namespace)
    first_label = namespace["SMOKE_LABEL"]
    assert first_label == "notebook-smoke-existing-smoke-session"
    # Simulate enabling the pilot after a smoke run, rather than restarting it.
    exec(settings.replace("RUN_PILOT = False", "RUN_PILOT = True"), namespace)
    assert namespace["SMOKE_LABEL"] == first_label and namespace["RUN_PILOT"] is True


def test_preflight_cli_selects_only_the_requested_task(monkeypatch):
    from rollout_cost import cli

    seen = {}

    def fake_run(cfg, split, runs_dir, label=None):
        from rollout_cost.fixtures import load_split
        seen["tasks"] = [task.task_id for task in load_split(cfg, split)]
        return Path("unused"), {}

    monkeypatch.setattr(cli, "run_sync", fake_run)
    monkeypatch.setattr(cli, "_print_summary", lambda *args: None)
    assert cli.main(["run", "--config", str(ROOT / "configs/synthetic.yaml"), "--task", "config_merge"]) == 0
    assert seen["tasks"] == ["config_merge"]


def test_notebook_profile_through_mock_server_and_archive(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    assert main(["pin", "--profile", "notebook", "--rehearsal", "--configs-dir", str(configs), "--cpus", "0,1"]) == 0
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = load_config(configs / "notebook-smoke.pinned.yaml", {
        "model": {"base_url": f"http://127.0.0.1:{port}/v1"},
        "server": {"args": {"port": port, "base-latency": 0.01, "per-token-latency": 0.0}}})
    cfg.root = ROOT
    path, summary = asyncio.run(execute_run(cfg, "dev", tmp_path / "runs", label="notebook-test", run_id="mock"))
    assert summary["cohort_size"] == 6 and summary["outcomes"]["infra_error"] == 0
    assert summary["successes"] == 4 and summary["drain"]["server_idle"]["verified"]
    assert latest_run(tmp_path, "notebook-test") == path
    with pytest.raises(RuntimeError, match="real_gpu_run"):
        check_result(path)  # A successful rehearsal must never pass as a GPU test.
    with tarfile.open(archive_results(tmp_path)) as archive:
        assert "runs/mock/summary.json" in archive.getnames()


def test_quality_gate_accepts_only_a_healthy_real_run(tmp_path):
    manifest = {"status": "complete", "synthetic": False, "gpu": {"available": True}}
    summary = {"outcomes": {"infra_error": 0}, "successes": 1, "protocol_error_rate": 0.0,
               "wall_timeouts": 0, "drain": {"server_idle": {"verified": True}}, "window_s": 1}
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert check_result(tmp_path)["successes"] == 1
    broken = deepcopy(summary)
    broken["successes"] = 0
    (tmp_path / "summary.json").write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="at_least_one_success"):
        check_result(tmp_path)
