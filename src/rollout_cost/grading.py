"""Independent grading of a final workspace with runner-only hidden tests.

The grading directory is built from the pristine fixture (public tests, pytest
configuration) with only the task's ``graded_paths`` taken from the agent's
workspace, plus the hidden tests. Editing public tests therefore cannot make a
task pass, and hidden tests are never present in a workspace the agent sees.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .fixtures import Task
from .tools import ExecutorError, ToolPool, copy_tree, truncate

GRADER_COMMAND = ("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "tests_hidden")


@dataclass
class GradeResult:
    outcome: str  # pass | fail | grader_error
    detail: str | None
    exit_code: int | None
    duration_s: float
    output_tail: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_grade_dir(task: Task, source_root: Path, grade_dir: Path) -> None:
    if grade_dir.exists():
        shutil.rmtree(grade_dir)
    copy_tree(task.workspace_dir, grade_dir)
    for rel in task.graded_paths:
        target = grade_dir / rel
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        source = source_root / rel
        if source.is_dir() and not source.is_symlink():
            copy_tree(source, target)
        elif source.is_file() and not source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    copy_tree(task.grader_dir / "hidden", grade_dir / "tests_hidden")


def classify(exit_code: int | None, timed_out: bool) -> tuple[str, str | None]:
    if timed_out:
        # Hidden tests hang when the submitted code does; the timeout is generous.
        return "fail", "grader_timeout"
    if exit_code == 0:
        return "pass", None
    if exit_code in (1, 2):  # test failures, or collection errors from broken code
        return "fail", f"pytest_exit_{exit_code}"
    return "grader_error", f"pytest_exit_{exit_code}"


async def grade(task: Task, source_root: Path, grade_dir: Path, executor, pool: ToolPool, timeout_s: float, label: str) -> GradeResult:
    started = time.monotonic()
    try:
        build_grade_dir(task, source_root, grade_dir)
    except OSError as exc:
        return GradeResult("grader_error", f"prepare_failed: {exc}", None, time.monotonic() - started, "")
    slot = await pool.acquire()
    try:
        result = await executor.run(grade_dir, GRADER_COMMAND, timeout_s, slot, label)
    except ExecutorError as exc:
        return GradeResult("grader_error", f"executor: {exc}", None, time.monotonic() - started, "")
    finally:
        pool.release(slot)
    outcome, detail = classify(result.exit_code, result.timed_out)
    tail, _ = truncate(result.output, 4000, head_fraction=0.0)
    return GradeResult(outcome, detail, result.exit_code, time.monotonic() - started, tail)


async def validate_task(task: Task, scratch: Path, executor, pool: ToolPool, timeout_s: float) -> dict:
    """Check the starting code fails and the reference fix passes, for public and hidden tests."""
    checks = {}
    reference = task.grader_dir / "reference"
    for name, source in (("starting", task.workspace_dir), ("reference", reference)):
        result = await grade(task, source, scratch / f"{task.task_id}-{name}", executor, pool, timeout_s, f"validate-{name}")
        checks[f"{name}_hidden"] = result.outcome
        ws = scratch / f"{task.task_id}-{name}-public"
        if ws.exists():
            shutil.rmtree(ws)
        copy_tree(task.workspace_dir, ws)
        if name == "reference":
            copy_tree(reference, ws)
        slot = await pool.acquire()
        try:
            public = await executor.run(ws, task.test_command, timeout_s, slot, f"validate-{name}-public")
        finally:
            pool.release(slot)
        checks[f"{name}_public"] = classify(public.exit_code, public.timed_out)[0]
    ok = (
        checks["starting_hidden"] == "fail"
        and checks["starting_public"] == "fail"
        and checks["reference_hidden"] == "pass"
        and checks["reference_public"] == "pass"
    )
    return {"task_id": task.task_id, "ok": ok, **checks}
