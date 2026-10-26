"""Workspace file operations, the tool-worker pool and command executors.

File actions (read/edit/write) are performed by the trusted runner on the host
after path validation. Only test commands execute generated code, and they run
through an executor:

* ``DockerExecutor``: no network, non-root, read-only root filesystem and
  read-only workspace mount, CPU pinned to the tool slot, memory/pid limits.
* ``LocalExecutor``: a plain subprocess with a scrubbed environment. It provides
  no isolation and is for development with trusted (scripted) models only.

The ``ToolPool`` fixes the aggregate CPU budget for tool work: there are exactly
``len(worker_cpus)`` slots no matter how many episodes are active, and each
slot is pinned to one CPU. Episodes wait for a slot rather than oversubscribing
the host the model server shares.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from .fixtures import IGNORED_NAMES


class ToolError(Exception):
    """A tool request the model got wrong (bad path, missing text). Fed back to the model."""


class ExecutorError(Exception):
    """The execution environment itself failed. Recorded as an infrastructure error."""


# ---------------------------------------------------------------------------
# Paths and file actions


def resolve_workspace_path(root: Path, rel: str) -> Path:
    if not rel or "\x00" in rel or "\\" in rel:
        raise ToolError(f"Invalid path {rel!r}: use forward slashes and a relative path.")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ":" in pure.parts[0] or ".." in pure.parts:
        raise ToolError(f"Invalid path {rel!r}: paths must stay inside the repository.")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ToolError(f"Invalid path {rel!r}: symbolic links are not allowed.")
    resolved_root = root.resolve()
    target = current.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ToolError(f"Invalid path {rel!r}: paths must stay inside the repository.")
    return current


def truncate(text: str, max_chars: int, head_fraction: float) -> tuple[str, bool]:
    """Keep ``max_chars`` of ``text``: a head part and a tail part, with a marker."""
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * head_fraction)
    tail = max_chars - head
    omitted = len(text) - max_chars
    marker = f"\n[... {omitted} characters omitted ...]\n"
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def read_file(root: Path, rel: str, max_chars: int) -> tuple[str, bool]:
    path = resolve_workspace_path(root, rel)
    if not path.is_file():
        raise ToolError(f"File not found: {rel}")
    return truncate(path.read_bytes().decode("utf-8", errors="replace"), max_chars, head_fraction=1.0)


def write_file(root: Path, rel: str, content: str) -> int:
    path = resolve_workspace_path(root, rel)
    if path.is_dir():
        raise ToolError(f"{rel} is a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    path.write_bytes(data)
    return len(data)


def edit_file(root: Path, rel: str, old: str, new: str) -> None:
    path = resolve_workspace_path(root, rel)
    if not path.is_file():
        raise ToolError(f"File not found: {rel}")
    if not old:
        raise ToolError("edit_file requires a non-empty \"old\" string.")
    text = path.read_bytes().decode("utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        raise ToolError(f"The \"old\" text was not found in {rel}. It must match the file exactly, including whitespace.")
    if count > 1:
        raise ToolError(f"The \"old\" text occurs {count} times in {rel}. Include more context so it matches once.")
    path.write_bytes(text.replace(old, new, 1).encode("utf-8"))


def list_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not (IGNORED_NAMES & set(p.relative_to(root).parts))
    )


def _skip_symlinks_and_caches(directory: str, names: list[str]) -> list[str]:
    return [n for n in names if n in IGNORED_NAMES or os.path.islink(os.path.join(directory, n))]


def copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory without following or copying symlinks."""
    shutil.copytree(src, dst, ignore=_skip_symlinks_and_caches, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Tool-worker pool


@dataclass(frozen=True)
class ToolSlot:
    index: int
    cpu: int | None


class ToolPool:
    """FIFO pool of tool slots; each slot is one pinned CPU when cpus are given."""

    def __init__(self, cpus: Sequence[int] | None, workers: int):
        slots = [ToolSlot(i, cpu) for i, cpu in enumerate(cpus)] if cpus else [ToolSlot(i, None) for i in range(workers)]
        self.size = len(slots)
        self._free: asyncio.Queue[ToolSlot] = asyncio.Queue()
        for slot in slots:
            self._free.put_nowait(slot)
        self.in_use = 0
        self.max_in_use = 0
        self.waiting = 0

    async def acquire(self) -> ToolSlot:
        self.waiting += 1
        try:
            slot = await self._free.get()
        finally:
            self.waiting -= 1
        self.in_use += 1
        self.max_in_use = max(self.max_in_use, self.in_use)
        return slot

    def release(self, slot: ToolSlot) -> None:
        self.in_use -= 1
        self._free.put_nowait(slot)


# ---------------------------------------------------------------------------
# Executors


@dataclass
class ProcessResult:
    exit_code: int | None
    output: str
    timed_out: bool
    duration_s: float


OUTPUT_LIMIT_BYTES = 1_000_000


def _decode(data: bytes) -> str:
    return data[:OUTPUT_LIMIT_BYTES].decode("utf-8", errors="replace").replace("\r\n", "\n")


def cpu_pinning_supported() -> bool:
    return hasattr(os, "sched_setaffinity")


class LocalExecutor:
    """Runs commands as host subprocesses. No container isolation.

    On Linux each process is pinned to its tool slot's CPU, so the fixed
    aggregate tool CPU budget holds even without Docker (e.g. a Lightning AI
    Studio). On other platforms slots bound concurrency but are not pinned.
    """

    name = "local"
    isolation = "none (no container; generated code runs as the runner's user)"

    def __init__(self) -> None:
        self._live: set[asyncio.subprocess.Process] = set()

    def describe(self) -> dict:
        return {"name": self.name, "isolation": self.isolation, "python": "python", "python_version": sys.version.split()[0],
                "cpu_pinning": cpu_pinning_supported()}

    def _env(self) -> dict[str, str]:
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
        for key in ("SYSTEMROOT", "TEMP", "TMP", "TMPDIR"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _kill(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    async def run(self, workdir: Path, command: Sequence[str], timeout_s: float, slot: ToolSlot, label: str) -> ProcessResult:
        argv = [sys.executable if part == "python" else part for part in command]
        if slot.cpu is not None and cpu_pinning_supported():
            taskset = shutil.which("taskset")
            if taskset is None:
                raise ExecutorError("CPU-pinned local execution requires taskset (util-linux).")
            # Apply affinity in an executable, not a Python preexec_fn: the live
            # exporter and subprocess watchers may already have started threads.
            argv = [taskset, "--cpu-list", str(slot.cpu), *argv]
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workdir),
                env=self._env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **kwargs,
            )
        except OSError as exc:
            raise ExecutorError(f"could not start {argv[0]}: {exc}") from exc
        self._live.add(proc)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
            return ProcessResult(proc.returncode, _decode(out), False, time.monotonic() - started)
        except TimeoutError:
            self._kill(proc)
            await proc.wait()
            return ProcessResult(None, "", True, time.monotonic() - started)
        finally:
            # Also reached when the episode deadline cancels us.
            self._kill(proc)
            self._live.discard(proc)

    async def cleanup(self) -> list[str]:
        leftovers = [str(p.pid) for p in self._live if p.returncode is None]
        for proc in list(self._live):
            self._kill(proc)
        self._live.clear()
        return leftovers


class DockerExecutor:
    """Runs each command in a disposable, network-less, resource-limited container."""

    name = "docker"
    # Exit codes produced by `docker run` itself rather than the command.
    DOCKER_FAILURE_CODES = frozenset({125, 126, 127})

    def __init__(self, image: str, run_id: str, memory_mb: int, pids_limit: int, docker: str = "docker"):
        self.image = image
        self.run_id = run_id
        self.memory_mb = memory_mb
        self.pids_limit = pids_limit
        self.docker = docker
        self._counter = itertools.count()
        self.isolation = "container: no network, non-root, read-only rootfs and workspace, pinned CPU, memory/pid limits"

    def describe(self) -> dict:
        image_id = None
        try:
            proc = subprocess.run(
                [self.docker, "image", "inspect", "--format", "{{.Id}}", self.image], capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0:
                image_id = proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {"name": self.name, "isolation": self.isolation, "image": self.image, "image_id": image_id,
                "memory_mb": self.memory_mb, "pids_limit": self.pids_limit}

    def command(self, container: str, workdir: Path, command: Sequence[str], slot: ToolSlot) -> list[str]:
        argv = [
            self.docker, "run", "--rm", "--name", container,
            "--label", f"rollout_cost.run_id={self.run_id}",
            "--network", "none",
            "--user", "65534:65534",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
            "--pids-limit", str(self.pids_limit),
            "--cpus", "1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
            "-v", f"{workdir.resolve()}:/workspace:ro", "-w", "/workspace",
        ]
        if slot.cpu is not None:
            argv += ["--cpuset-cpus", str(slot.cpu)]
        return [*argv, self.image, *command]

    def _kill(self, container: str) -> None:
        try:
            subprocess.run([self.docker, "kill", container], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass

    async def run(self, workdir: Path, command: Sequence[str], timeout_s: float, slot: ToolSlot, label: str) -> ProcessResult:
        container = f"rc-{self.run_id}-{label}-{next(self._counter)}"[:120]
        argv = self.command(container, workdir, command, slot)
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        except OSError as exc:
            raise ExecutorError(f"could not run docker: {exc}") from exc
        finished = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
            finished = True
        except TimeoutError:
            self._kill(container)
            await proc.wait()
            return ProcessResult(None, "", True, time.monotonic() - started)
        finally:
            if not finished:
                self._kill(container)
        if proc.returncode in self.DOCKER_FAILURE_CODES:
            raise ExecutorError(f"docker run failed with exit code {proc.returncode}: {_decode(out)[-2000:]}")
        return ProcessResult(proc.returncode, _decode(out), False, time.monotonic() - started)

    async def cleanup(self) -> list[str]:
        """Kill any container still labelled with this run; return their ids."""
        try:
            proc = subprocess.run(
                [self.docker, "ps", "-q", "--filter", f"label=rollout_cost.run_id={self.run_id}"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ["<docker ps failed>"]
        leftovers = proc.stdout.split()
        for container in leftovers:
            self._kill(container)
        return leftovers


def make_executor(kind: str, image: str, run_id: str, memory_mb: int, pids_limit: int):
    if kind == "local":
        return LocalExecutor()
    if kind == "docker":
        return DockerExecutor(image, run_id, memory_mb, pids_limit)
    raise ValueError(f"unknown executor {kind!r}")
