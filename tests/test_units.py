"""Unit tests for accounting, outcomes, protocol, paths, admission and config."""

import asyncio
import os
from pathlib import Path

import pytest

from rollout_cost.admission import FifoSemaphore
from rollout_cost.analysis import time_at_cap
from rollout_cost.config import ConfigError, deep_merge, load_config, protected_settings, render_args
from rollout_cost.cost import allocation_cost, billed_seconds, per_unit
from rollout_cost.outcomes import derive_category
from rollout_cost.protocol import ProtocolError, parse_action
from rollout_cost.tools import ToolError, edit_file, python_syntax_feedback, resolve_workspace_path, truncate

ROOT = Path(__file__).resolve().parents[1]


# -- accounting --------------------------------------------------------------


def test_billing_rounding():
    assert billed_seconds(61, 60) == 120
    assert billed_seconds(60, 60) == 60
    assert billed_seconds(0.5, None) == 0.5
    assert allocation_cost("x", 1800, 2.0, "USD", 3600).cost == pytest.approx(2.0)


def test_unknown_price_is_unknown_not_zero():
    line = allocation_cost("x", 100, None, "USD")
    assert line.cost is None
    assert per_unit(line.cost, 10) is None


def test_zero_successes_has_undefined_cost_per_success():
    assert per_unit(1.0, 0) is None
    assert per_unit(1.0, 4) == 0.25


def test_illustrative_plan_example():
    # 600 s at $1.50/h = $0.25; 450 s = $0.1875 (plan section 8).
    assert allocation_cost("b", 600, 1.5, "USD").cost == pytest.approx(0.25)
    assert allocation_cost("c", 450, 1.5, "USD").cost == pytest.approx(0.1875)


def test_time_at_cap():
    # Two overlapping intervals inside [0, 10]: both active during [2, 5].
    assert time_at_cap([(0, 5), (2, 8)], 2, 0, 10) == pytest.approx(0.3)
    assert time_at_cap([(0, 10)], 1, 0, 10) == pytest.approx(1.0)


def _comparison(base_window, cand_window, varied=True):
    def headline(window):
        return {"window_s": window, "cost": None, "successes": 5, "valid_episodes": 6, "infra_errors": 0,
                "wall_timeouts": 0, "service_p90_s": 10.0, "cohort_size": 6}

    return {"problems": [], "varied": {"execution.max_active_episodes": {}} if varied else {},
            "per_task": {"t": {"baseline": 5, "candidate": 5, "episodes": 6}},
            "baseline_summary": headline(base_window), "candidate_summary": headline(cand_window)}


def test_acceptance_requires_reduction_above_trial_spread():
    from rollout_cost.analysis import evaluate_acceptance

    # 10% pooled reduction, but baseline windows vary by ~18% between trials.
    noisy = evaluate_acceptance([_comparison(100, 90), _comparison(120, 108)])
    assert not noisy["accepted"]
    assert {c["check"]: c["ok"] for c in noisy["checks"]}["reduction exceeds trial-to-trial spread"] is False
    stable = evaluate_acceptance([_comparison(100, 80), _comparison(102, 82)])
    assert stable["accepted"], stable["checks"]


def test_identical_configs_are_an_aa_calibration_not_a_decision():
    from rollout_cost.analysis import evaluate_acceptance

    result = evaluate_acceptance([_comparison(100, 80, varied=False), _comparison(101, 81, varied=False)])
    assert result["aa_calibration"] and not result["accepted"]


def test_single_trial_cannot_be_accepted_by_default():
    from rollout_cost.analysis import evaluate_acceptance

    assert not evaluate_acceptance([_comparison(100, 50)])["accepted"]
    relaxed = evaluate_acceptance([_comparison(100, 50)], {"min_trials": 1, "require_reduction_above_spread": False})
    assert relaxed["accepted"]


# -- outcome fields ----------------------------------------------------------


@pytest.mark.parametrize(
    "stop, infra, grade, expected",
    [
        ("finished", "ok", "pass", "success"),
        ("max_steps", "ok", "pass", "success"),  # correct patch, budget exhausted
        ("finished", "ok", "fail", "failed_after_finish"),
        ("max_steps", "ok", "fail", "failed_budget_exhausted"),
        ("context_exhausted", "ok", "fail", "failed_budget_exhausted"),
        ("wall_timeout", "ok", "fail", "failed_budget_exhausted"),
        ("infra_error", "model_server_error", "pass", "infra_error"),  # never a model-quality signal
        ("finished", "ok", "grader_error", "infra_error"),
        ("finished", "ok", "not_graded", "infra_error"),
    ],
)
def test_derive_category(stop, infra, grade, expected):
    assert derive_category(stop, infra, grade) == expected


# -- protocol ----------------------------------------------------------------


def test_parse_plain_and_fenced_actions():
    assert parse_action('{"action": "run_tests"}', "stop", 1000).name == "run_tests"
    fenced = 'Sure:\n```json\n{"action": "read_file", "path": "src/a.py"}\n```'
    action = parse_action(fenced, "stop", 1000)
    assert action.args == {"path": "src/a.py"}
    assert action.extra_text_chars > 0


def test_truncated_reply_is_rejected_even_if_json_decodes():
    with pytest.raises(ProtocolError) as info:
        parse_action('{"action": "finish"}', "length", 1000)
    assert info.value.kind == "truncated"


@pytest.mark.parametrize(
    "text, kind",
    [
        ("no json here", "no_json"),
        ('{"action": "write_file", "path": "a.py", "content": "x', "invalid_json"),
        ('{"action": "delete_everything"}', "unknown_action"),
        ('{"action": "read_file"}', "bad_arguments"),
        ('{"action": "edit_file", "path": "a", "old": 1, "new": "x"}', "bad_arguments"),
        ('{"action": "write_file", "path": "a", "content": "' + "x" * 50 + '"}', "payload_too_large"),
    ],
)
def test_protocol_errors(text, kind):
    with pytest.raises(ProtocolError) as info:
        parse_action(text, "stop", 20)
    assert info.value.kind == kind


# -- workspace paths ---------------------------------------------------------


@pytest.mark.parametrize("rel", ["../outside.py", "/etc/passwd", "src\\a.py", "C:/x.py", "", "src/../../x"])
def test_paths_cannot_escape_workspace(tmp_path, rel):
    with pytest.raises(ToolError):
        resolve_workspace_path(tmp_path, rel)


def test_symlinks_are_rejected(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    ws = tmp_path / "ws"
    ws.mkdir()
    try:
        os.symlink(outside, ws / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform")
    with pytest.raises(ToolError):
        resolve_workspace_path(ws, "link.txt")


def test_edit_file_requires_unique_match(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\nx = 1\n")
    with pytest.raises(ToolError, match="2 times"):
        edit_file(tmp_path, "a.py", "x = 1", "x = 2")
    with pytest.raises(ToolError, match="not found"):
        edit_file(tmp_path, "a.py", "y = 1", "y = 2")
    edit_file(tmp_path, "a.py", "x = 1\nx", "x = 1\ny")
    assert (tmp_path / "a.py").read_bytes() == b"x = 1\ny = 1\n"


def test_identical_edit_reports_no_change_and_preserves_file(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"x = 1\n")
    with pytest.raises(ToolError, match="identical"):
        edit_file(tmp_path, "a.py", "x = 1", "x = 1")
    assert path.read_bytes() == b"x = 1\n"


def test_python_syntax_feedback_preserves_invalid_edit_without_execution(tmp_path):
    path = tmp_path / "a.py"
    invalid = b"def f():\n    try:\n    return 1\n"
    path.write_bytes(invalid)
    feedback = python_syntax_feedback(tmp_path, "a.py")
    assert "syntax check failed" in feedback and "line 3" in feedback
    assert "2:     try:" in feedback and "3:     return 1" in feedback
    assert path.read_bytes() == invalid  # diagnosis never silently repairs a submission
    path.write_text("raise RuntimeError('must not execute')\n")
    assert "syntax check passed" in python_syntax_feedback(tmp_path, "a.py")
    path.write_text("continue\n")
    assert "not properly in loop" in python_syntax_feedback(tmp_path, "a.py")
    (tmp_path / "notes.txt").write_text("not Python")
    assert python_syntax_feedback(tmp_path, "notes.txt") == ""


def test_docker_command_isolation_flags(tmp_path):
    from rollout_cost.tools import DockerExecutor, ToolSlot

    argv = DockerExecutor("img:1", "run1", 512, 128).command("c1", tmp_path, ["python", "-m", "pytest"], ToolSlot(0, 5))
    joined = " ".join(argv)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--cpuset-cpus 5", "--user 65534:65534",
                 "--memory 512m", "--pids-limit 128", "--label rollout_cost.run_id=run1"):
        assert flag in joined
    mount = argv[argv.index("-v") + 1]
    assert mount.endswith(":/workspace:ro")
    assert argv[-4:] == ["img:1", "python", "-m", "pytest"]


def test_truncate_keeps_head_and_tail():
    text, cut = truncate("a" * 50 + "b" * 50, 20, head_fraction=0.2)
    assert cut and text.startswith("aaaa") and text.endswith("b" * 16) and "omitted" in text
    assert truncate("short", 20, 0.5) == ("short", False)


# -- admission ---------------------------------------------------------------


def test_fifo_semaphore_caps_and_order():
    async def go():
        sem = FifoSemaphore(2)
        order = []

        async def user(i):
            await sem.acquire()
            order.append(i)
            await asyncio.sleep(0.01)
            sem.release()

        await asyncio.gather(*(user(i) for i in range(8)))
        return sem.max_in_use, order

    max_in_use, order = asyncio.run(go())
    assert max_in_use == 2
    assert order == list(range(8))


def test_fifo_semaphore_cancelled_waiter_does_not_leak_slot():
    async def go():
        sem = FifoSemaphore(1)
        await sem.acquire()
        waiter = asyncio.create_task(sem.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        sem.release()
        await asyncio.wait_for(sem.acquire(), 1)  # slot must be available again
        return sem.in_use

    assert asyncio.run(go()) == 1


# -- config ------------------------------------------------------------------


def test_deep_merge_keeps_explicit_falsy_values():
    merged = deep_merge({"a": {"b": True, "c": 1}, "d": 5}, {"a": {"b": False}, "d": None})
    assert merged == {"a": {"b": False, "c": 1}, "d": None}


def test_extends_and_unknown_keys(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "base.yaml").write_text("execution: {max_active_episodes: 3}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nexecution: {max_inflight_llm_requests: 2}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert (cfg.execution.max_active_episodes, cfg.execution.max_inflight_llm_requests) == (3, 2)
    (tmp_path / "bad.yaml").write_text("execution: {max_active_episode: 3}\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(tmp_path / "bad.yaml")


def test_protected_settings_exclude_only_tunables():
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    protected = protected_settings(cfg.to_dict())
    assert "execution.max_active_episodes" not in protected
    assert "server.args.enable-prefix-caching" not in protected
    assert "workload.max_steps" in protected
    assert "tools.worker_cpus" in protected
    assert "server.args.max-model-len" in protected


def test_render_args():
    assert render_args({"max-model-len": 8192, "enable-prefix-caching": False, "x": None, "trust": True}) == [
        "--max-model-len", "8192", "--no-enable-prefix-caching", "--trust"]


def test_tool_cpus_must_not_overlap_server_cpus(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "c.yaml").write_text("tools: {worker_cpus: [2, 3]}\nserver: {reserved_cpus: [0, 1, 2]}\n")
    with pytest.raises(ConfigError, match="overlaps"):
        load_config(tmp_path / "c.yaml")
