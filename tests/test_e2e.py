"""End-to-end runs with a scripted model and the local executor (no GPU)."""

import asyncio
import json
from pathlib import Path


from rollout_cost.analysis import compare_runs, evaluate_acceptance, load_run
from rollout_cost.config import load_config
from rollout_cost.fixtures import load_split
from rollout_cost.model_client import ScriptedModelClient, ScriptedReply, build_scripts
from rollout_cost.runner import execute_run

ROOT = Path(__file__).resolve().parents[1]
FAST = {"scripted_base_latency_s": 0.01, "scripted_per_token_latency_s": 0.0, "scripted_contention": 0.0}


def config(**sections):
    overrides = {"model": dict(FAST), "workload": {"repetitions": 1}}
    for key, value in sections.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(ROOT / "configs" / "synthetic.yaml", overrides)


def run(cfg, tmp_path, client=None, name="run"):
    run_dir, summary = asyncio.run(execute_run(cfg, "dev", tmp_path, run_id=name, client=client))
    return load_run(run_dir), summary


def scripted(cfg, scripts, **kwargs):
    return ScriptedModelClient(scripts, base_latency_s=kwargs.get("latency", 0.01), per_token_latency_s=0.0, contention=0.0)


def test_reference_policy_solves_every_task_and_caps_hold(tmp_path):
    cfg = config(model={"scripted_policy": "reference"}, workload={"repetitions": 2},
                 execution={"max_active_episodes": 4, "max_inflight_llm_requests": 2}, tools={"workers": 2})
    tasks = load_split(cfg, "dev")
    client = scripted(cfg, build_scripts(tasks, "reference"))
    result, summary = run(cfg, tmp_path, client)
    assert summary["outcomes"]["success"] == 12
    assert summary["valid_episodes"] == 12
    assert client.max_inflight <= 2
    assert summary["max_inflight_observed"] <= 2
    assert summary["max_tool_slots_observed"] <= 2
    # Every submitted episode has exactly one terminal record and one terminal event.
    assert sorted(e["episode_id"] for e in result.episodes) == sorted(result.manifest["protected"]["cohort"])
    events = [json.loads(l) for l in (result.path / "events.jsonl").read_text().splitlines()]
    terminal = [e for e in events if e["event_type"] == "episode_terminal"]
    assert len(terminal) == 12 and len({e["episode_id"] for e in terminal}) == 12
    # Per-episode sequence numbers are contiguous.
    by_episode = {}
    for e in events:
        by_episode.setdefault(e["episode_id"], []).append(e["event_seq"])
    assert all(seqs == list(range(len(seqs))) for seqs in by_episode.values())


def test_noop_policy_fails_grading_without_infra_errors(tmp_path):
    cfg = config(model={"scripted_policy": "noop"})
    _, summary = run(cfg, tmp_path)
    assert summary["outcomes"] == {"success": 0, "failed_after_finish": 6, "failed_budget_exhausted": 0, "infra_error": 0}
    assert summary["cost"]["per_success"] is None


def test_editing_public_tests_does_not_count_as_a_fix(tmp_path):
    cfg = config()
    tasks = load_split(cfg, "dev")
    cheat = [
        ScriptedReply(json.dumps({"action": "write_file", "path": "tests/test_public.py", "content": "def test_ok():\n    pass\n"})),
        ScriptedReply(json.dumps({"action": "run_tests"})),
        ScriptedReply(json.dumps({"action": "finish"})),
    ]
    result, summary = run(cfg, tmp_path, scripted(cfg, {t.task_id: cheat for t in tasks}))
    assert summary["successes"] == 0
    transcript = json.loads((result.path / "transcripts" / f"{tasks[0].task_id}__r0.json").read_text())
    # [system, task, write, result, run_tests, result]: the tampered public tests passed.
    assert "exit code 0" in transcript["messages"][5]["content"]


def test_wall_timeout_cancels_outstanding_request(tmp_path):
    cfg = config(workload={"episode_wall_timeout_s": 0.5}, model={"request_timeout_s": 30})
    tasks = load_split(cfg, "dev")
    client = scripted(cfg, {t.task_id: [ScriptedReply("", hang=True)] for t in tasks})
    result, summary = run(cfg, tmp_path, client)
    assert summary["stop_reasons"] == {"wall_timeout": 6}
    assert summary["cancelled_requests"] == 6
    assert summary["outcomes"]["infra_error"] == 0  # a budget stop, not an infrastructure failure
    assert summary["outcomes"]["failed_budget_exhausted"] == 6
    assert client.inflight == 0
    assert any("wall-time backstop" in w for w in summary["warnings"])


def test_request_timeout_is_an_infrastructure_error(tmp_path):
    cfg = config(workload={"episode_wall_timeout_s": 30}, model={"request_timeout_s": 0.2})
    tasks = load_split(cfg, "dev")
    client = scripted(cfg, {t.task_id: [ScriptedReply("", hang=True)] for t in tasks})
    result, summary = run(cfg, tmp_path, client)
    assert summary["outcomes"]["infra_error"] == 6
    assert {e["infrastructure_status"] for e in result.episodes} == {"model_timeout"}


def test_server_error_is_infrastructure_but_still_graded(tmp_path):
    cfg = config()
    tasks = load_split(cfg, "dev")
    client = scripted(cfg, {t.task_id: [ScriptedReply("", error="server")] for t in tasks})
    result, summary = run(cfg, tmp_path, client)
    assert summary["outcomes"]["infra_error"] == 6
    assert {e["grader_outcome"] for e in result.episodes} == {"fail"}  # graded for diagnostics only


def test_truncated_reply_consumes_a_step(tmp_path):
    cfg = config(generation={"max_output_tokens_per_call": 64}, workload={"max_steps": 2})
    tasks = load_split(cfg, "dev")
    long_write = json.dumps({"action": "write_file", "path": "src/x.py", "content": "y" * 2000})
    client = scripted(cfg, {t.task_id: [ScriptedReply(long_write), ScriptedReply(long_write)] for t in tasks})
    result, summary = run(cfg, tmp_path, client)
    assert summary["protocol_errors"] == {"truncated": 12}
    assert summary["stop_reasons"] == {"max_steps": 6}
    assert not (result.path / "workspaces" / f"{tasks[0].task_id}__r0" / "src" / "x.py").exists()


def test_context_exhaustion_stops_before_overflowing(tmp_path):
    cfg = config(workload={"context_window_tokens": 1500, "max_steps": 6}, generation={"max_output_tokens_per_call": 200})
    tasks = load_split(cfg, "dev")
    reads = [ScriptedReply(json.dumps({"action": "read_file", "path": "tests/test_public.py"}))] * 6
    result, summary = run(cfg, tmp_path, scripted(cfg, {t.task_id: reads for t in tasks}))
    assert summary["stop_reasons"].get("context_exhausted", 0) == 6
    assert all(1 <= e["steps"] < 6 for e in result.episodes)
    budget = 1500 - 200
    for episode in result.episodes:
        transcript = json.loads((result.path / "transcripts" / f"{episode['episode_id']}.json").read_text())
        assert all(r["prompt_tokens"] <= budget for r in transcript["requests"])


def test_comparison_detects_task_swaps_hidden_by_equal_totals(tmp_path):
    cfg = config()
    tasks = load_split(cfg, "dev")
    ids = [t.task_id for t in tasks]
    solve = build_scripts(tasks, "reference")
    give_up = [ScriptedReply(json.dumps({"action": "finish"}))]
    # Baseline solves tasks 0-1, candidate solves tasks 1-2: same total, one regression.
    base_scripts = {t: (solve[t] if i in (0, 1) else give_up) for i, t in enumerate(ids)}
    cand_scripts = {t: (solve[t] if i in (1, 2) else give_up) for i, t in enumerate(ids)}
    base, _ = run(cfg, tmp_path, scripted(cfg, base_scripts), name="base")
    cand_cfg = config(execution={"max_active_episodes": 2})
    cand, _ = run(cand_cfg, tmp_path, scripted(cand_cfg, cand_scripts), name="cand")
    comp = compare_runs(base, cand)
    assert comp["valid"], comp["problems"]
    assert comp["varied"] == {"execution.max_active_episodes": {"baseline": 3, "candidate": 2}}
    assert comp["transitions"]["pass->fail"] == 1 and comp["transitions"]["fail->pass"] == 1
    acc = evaluate_acceptance([comp], {"min_window_reduction_pct": -1000, "require_reduction_every_pair": False})
    checks = {c["check"]: c["ok"] for c in acc["checks"]}
    assert checks["pooled successes"] is True
    assert checks["task-level regressions"] is False
    assert acc["accepted"] is False


def test_protected_setting_change_invalidates_comparison(tmp_path):
    base, _ = run(config(model={"scripted_policy": "noop"}), tmp_path, name="a")
    cand, _ = run(config(model={"scripted_policy": "noop"}, workload={"max_steps": 5}), tmp_path, name="b")
    comp = compare_runs(base, cand)
    assert not comp["valid"]
    assert any(d["key"] == "workload.max_steps" for d in comp["protected_diff"])
    assert evaluate_acceptance([comp])["accepted"] is False
