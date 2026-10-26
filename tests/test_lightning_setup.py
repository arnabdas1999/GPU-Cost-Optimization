"""Lightning setup checks without installing vLLM, accessing a GPU or downloading models."""

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rollout_cost.cli import main
from rollout_cost.config import load_config
from rollout_cost.tools import ExecutorError, LocalExecutor, ToolSlot

ROOT = Path(__file__).resolve().parents[1]


def test_local_cpu_pinning_uses_taskset_without_python_fork_callbacks(tmp_path, monkeypatch):
    from rollout_cost import tools

    launched = {}

    class Finished:
        returncode = 0

        async def communicate(self):
            return b"pinned\n", b""

    async def spawn(*argv, **kwargs):
        launched.update(argv=argv, kwargs=kwargs)
        return Finished()

    monkeypatch.setattr(tools, "cpu_pinning_supported", lambda: True)
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/taskset" if name == "taskset" else None)
    monkeypatch.setattr(tools.asyncio, "create_subprocess_exec", spawn)
    result = asyncio.run(LocalExecutor().run(tmp_path, ["python", "-c", "pass"], 2, ToolSlot(0, 3), "pin"))
    assert launched["argv"] == ("/usr/bin/taskset", "--cpu-list", "3", sys.executable, "-c", "pass")
    assert "preexec_fn" not in launched["kwargs"]
    assert result.exit_code == 0 and result.output == "pinned\n"


def test_missing_taskset_is_an_executor_error_not_unpinned_execution(tmp_path, monkeypatch):
    from rollout_cost import tools

    monkeypatch.setattr(tools, "cpu_pinning_supported", lambda: True)
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    with pytest.raises(ExecutorError, match="taskset"):
        asyncio.run(LocalExecutor().run(tmp_path, ["python", "-c", "pass"], 2, ToolSlot(0, 3), "pin"))


def test_pin_cli_keeps_model_download_and_serving_repository_consistent(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    assert main(["pin", "--configs-dir", str(configs), "--model-repo", "example/other-coder",
                 "--revision", "resolved-sha", "--vllm", "/opt/vllm/bin/vllm", "--cpus", "0,1,2,3"]) == 0
    for name in ("lightning.pinned.yaml", "smoke.pinned.yaml"):
        cfg = load_config(configs / name)
        assert cfg.model.repository == "example/other-coder"
        assert cfg.model.revision == cfg.model.tokenizer_revision == "resolved-sha"
        command = cfg.server.start_command
        assert command[command.index("serve") + 1] == cfg.model.repository
        assert cfg.server.args["revision"] == cfg.server.args["tokenizer-revision"] == cfg.model.revision


@pytest.fixture
def setup_host(tmp_path):
    # Prefer Git Bash on Windows; the system bash.exe may be a WSL launcher.
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is needed to exercise the setup script")

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/lightning-setup.sh", project / "scripts/lightning-setup.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    venv = tmp_path / "vllm-venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "commands.log"
    version = tmp_path / "installed-version"
    version.write_text("1.0.0\n")

    def executable(path, source):
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + source, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    executable(venv / "bin/python", '''
printf 'python %s\n' "$*" >> "$SETUP_TEST_LOG"
if [[ "$*" == *importlib.metadata* ]]; then cat "$SETUP_TEST_VERSION_FILE"; fi
''')
    executable(venv / "bin/vllm", '''
printf 'vllm %s\n' "$*" >> "$SETUP_TEST_LOG"
echo 'mock vllm help'
''')
    executable(fake_bin / "uv", '''
printf 'uv %s\n' "$*" >> "$SETUP_TEST_LOG"
if [[ "${1:-}" == pip && "${2:-}" == install ]]; then
  if [[ "${SETUP_TEST_KEEP_VERSION:-0}" != 1 ]]; then
    printf '%s\n' "$VLLM_VERSION" > "$SETUP_TEST_VERSION_FILE"
  fi
elif [[ "${1:-}" == pip && "${2:-}" == freeze ]]; then
  printf 'vllm==%s\n' "$VLLM_VERSION"
elif [[ "${1:-}" == run && "$*" == *HfApi* ]]; then
  echo resolved-sha
fi
''')
    for name in ("nvidia-smi", "docker"):
        executable(fake_bin / name, f'printf \'{name} %s\\n\' "$*" >> "$SETUP_TEST_LOG"\n')

    env = os.environ.copy()
    paths = [str(fake_bin)]
    if os.name == "nt" and git_bash.exists():
        paths.append(str(git_bash.parents[1] / "usr/bin"))
    paths.append(env.get("PATH", ""))
    env.update(PATH=os.pathsep.join(paths), VLLM_VENV=venv.as_posix(), VLLM_VERSION="2.0.0",
               MODEL_REPO="example/selected-coder", SETUP_TEST_LOG=log.as_posix(),
               SETUP_TEST_VERSION_FILE=version.as_posix())

    def run(*args, keep_wrong_version=False):
        child_env = {**env, "SETUP_TEST_KEEP_VERSION": "1" if keep_wrong_version else "0"}
        result = subprocess.run([bash, "--noprofile", "--norc", "scripts/lightning-setup.sh", *args],
                                cwd=project, env=child_env, capture_output=True, text=True, timeout=20)
        commands = log.read_text() if log.exists() else ""
        return result, commands

    return run, project, version


def test_setup_upgrades_existing_vllm_and_pins_selected_model(setup_host):
    run, project, version = setup_host
    result, commands = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "vllm==2.0.0" in commands  # install is not skipped just because the executable exists
    assert version.read_text().strip() == "2.0.0"
    assert "--model-repo example/selected-coder --revision resolved-sha" in commands
    assert "doctor --config configs/lightning.pinned.yaml --start-server" in commands
    assert (project / "vllm-packages.txt").read_text().strip() == "vllm==2.0.0"


def test_setup_rejects_a_version_mismatch_before_downloading_the_model(setup_host):
    run, _, _ = setup_host
    result, commands = run(keep_wrong_version=True)
    assert result.returncode != 0
    assert "requested 2.0.0, installed 1.0.0" in result.stderr
    assert "snapshot_download" not in commands
    assert "rollout-cost pin" not in commands


def test_prepare_downloads_on_cpu_without_starting_gpu_services(setup_host):
    run, project, _ = setup_host
    result, commands = run("--prepare")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_download" in commands and "example/selected-coder resolved-sha" in commands
    assert (project / "vllm-packages.txt").exists()
    for gpu_step in ("nvidia-smi", "docker", "import torch", "vllm serve", "rollout-cost pin", "doctor"):
        assert gpu_step not in commands


def test_rehearsal_does_not_install_vllm_or_download_a_model(setup_host):
    run, _, _ = setup_host
    result, commands = run("--rehearsal")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--rehearsal" in commands and "doctor --config" in commands
    for real_step in ("nvidia-smi", "pip install", "HfApi", "snapshot_download"):
        assert real_step not in commands
