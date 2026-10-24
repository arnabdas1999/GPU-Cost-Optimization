"""Task fixtures, content hashes and cohort construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Config, ConfigError

IGNORED_NAMES = frozenset({"__pycache__", ".pytest_cache"})
DEFAULT_TEST_COMMAND = ("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    title: str
    description: str
    initial_files: tuple[str, ...]
    graded_paths: tuple[str, ...]
    test_command: tuple[str, ...]
    workspace_dir: Path
    grader_dir: Path
    fixture_hash: str
    grader_hash: str


@dataclass(frozen=True)
class EpisodeSpec:
    index: int
    episode_id: str
    task: Task
    rep: int
    seed: int


def hash_tree(root: Path) -> str:
    """Stable content hash of a directory (relative paths and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if IGNORED_NAMES & set(rel.parts):
            continue
        digest.update(rel.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_task(task_dir: Path, graders_dir: Path) -> Task:
    meta = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    task_id = meta["id"]
    if task_id != task_dir.name:
        raise ConfigError(f"task id {task_id!r} does not match directory {task_dir.name!r}")
    workspace = task_dir / "workspace"
    grader = graders_dir / task_id
    for required in (workspace, grader / "hidden", grader / "reference"):
        if not required.is_dir():
            raise ConfigError(f"task {task_id}: missing {required}")
    return Task(
        task_id=task_id,
        family=meta["family"],
        title=meta["title"],
        description=meta["description"].strip(),
        initial_files=tuple(meta.get("initial_files", ())),
        graded_paths=tuple(meta.get("graded_paths", ("src",))),
        test_command=tuple(meta.get("test_command", DEFAULT_TEST_COMMAND)),
        workspace_dir=workspace,
        grader_dir=grader,
        fixture_hash=hash_tree(task_dir),
        grader_hash=hash_tree(grader),
    )


def load_split(cfg: Config, split: str) -> list[Task]:
    splits = yaml.safe_load(cfg.path(cfg.workload.splits_file).read_text(encoding="utf-8")) or {}
    if split not in splits:
        raise ConfigError(f"unknown split {split!r}; available: {sorted(splits)}")
    task_ids = splits[split] or []
    if not task_ids:
        raise ConfigError(f"split {split!r} has no tasks")
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for task_id in ids or []:
            if task_id in seen and seen[task_id] != name:
                raise ConfigError(f"task {task_id!r} appears in splits {seen[task_id]!r} and {name!r}")
            seen[task_id] = name
    fixtures_dir = cfg.path(cfg.workload.fixtures_dir)
    graders_dir = cfg.path(cfg.workload.graders_dir)
    return [load_task(fixtures_dir / task_id, graders_dir) for task_id in task_ids]


def episode_seed(base_seed: int, task_id: str, rep: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task_id}:{rep}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def build_cohort(tasks: list[Task], repetitions: int, base_seed: int) -> list[EpisodeSpec]:
    """Repetition-major order: every task once, then every task again, and so on.

    The order is fixed for every configuration, and admission is strictly FIFO.
    """
    cohort: list[EpisodeSpec] = []
    for rep in range(repetitions):
        for task in tasks:
            cohort.append(
                EpisodeSpec(
                    index=len(cohort),
                    episode_id=f"{task.task_id}__r{rep}",
                    task=task,
                    rep=rep,
                    seed=episode_seed(base_seed, task.task_id, rep),
                )
            )
    return cohort
