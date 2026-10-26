"""Standard-library helpers for the Colab/Kaggle notebook kernel.

The kernel does not need the runner or vLLM dependencies installed in its own
environment. All measured work is launched through the project's uv environment.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile


def check_gpu(root: Path) -> None:
    """Fail before large downloads on CPU/P100 sessions; record the assigned GPU."""
    command = ["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,compute_cap,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Select a GPU notebook runtime, then rerun this cell. "
                           "On Kaggle choose T4, and enable Internet.") from exc
    fields = next(csv.reader(result.stdout.strip().splitlines()))
    name, memory, capability, driver = (value.strip() for value in fields)
    hardware = (f"GPU 0: {name}; VRAM: {memory} MiB; compute capability: {capability}; driver: {driver}\n"
                f"Kernel Python: {sys.version}\n"
                f"Available CPUs: {sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}\n")
    (root / "notebook-hardware.txt").write_text(hardware, encoding="utf-8")
    print(hardware)
    if float(capability) < 7.5:
        raise RuntimeError(f"{name} is unsupported by the selected vLLM release. "
                           "Choose a T4 GPU runtime; Kaggle P100 is unsuitable.")


def run_logged(root: Path, command: list[str], log_name: str, env: dict | None = None) -> None:
    """Stream output into the notebook and a log; allow CLI cleanup on Stop."""
    print("Running:", " ".join(command), flush=True)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    with (root / log_name).open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=root, env=child_env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.wait(timeout=10)
            raise
        finally:
            proc.stdout.close()
    if code:
        raise RuntimeError(f"Command exited with code {code}; inspect {root / log_name}. "
                           "Run the archive cell to save diagnostics.")


def latest_run(root: Path, label: str) -> Path:
    candidates = []
    for path in (root / "runs").glob("*/run.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("label") == label:
            candidates.append((manifest["created_utc"], path.parent))
    if not candidates:
        raise RuntimeError(f"No run with label {label!r}; check the command log.")
    return max(candidates)[1]


def check_result(run_dir: Path) -> dict:
    """Require a real completed run and basic quality before longer experiments."""
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checks = {
        "complete": manifest.get("status") == "complete",
        "real_gpu_run": not manifest["synthetic"] and manifest["gpu"]["available"],
        "no_infrastructure_errors": summary["outcomes"]["infra_error"] == 0,
        "at_least_one_success": summary["successes"] >= 1,
        "protocol_errors_below_20_percent": (summary["protocol_error_rate"] or 0) < 0.20,
        "no_wall_timeouts": summary["wall_timeouts"] == 0,
        "drain_verified": (summary.get("drain") or {}).get("server_idle", {}).get("verified") is True,
    }
    cfg = manifest.get("config") or {}
    workload = cfg.get("workload") or {}
    print(json.dumps({"run": str(run_dir), "model": (cfg.get("model") or {}).get("repository"),
                      "edit_mode": workload.get("edit_mode", "literal"),
                      "budgets": {key: workload.get(key) for key in ("max_steps", "context_window_tokens")},
                      "stop_reasons": summary.get("stop_reasons"),
                      "checks": checks, "outcomes": summary["outcomes"],
                      "window_s": summary["window_s"]}, indent=2))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Check the report/transcripts before continuing: {', '.join(failed)}. "
                           "Run the archive cell to save results even after this error.")
    return summary


def archive_results(root: Path) -> Path:
    """Save results/configs/diagnostics, excluding packages, model caches and secrets."""
    destination = root / "notebook-results.tgz"
    paths = [root / name for name in ("runs", "experiments", "reports", "configs", "uv.lock",
                                     "vllm-packages.txt", "vllm-serve-help.txt", "notebook-hardware.txt")]
    paths.extend(root.glob("notebook-*.log"))
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            if path.exists():
                archive.add(path, arcname=path.name)
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB)")
    return destination
