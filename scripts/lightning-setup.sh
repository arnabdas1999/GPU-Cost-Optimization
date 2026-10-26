#!/usr/bin/env bash
# Prepare a Lightning AI Studio (or any Linux GPU host) for a measured session.
#
#   VLLM_VERSION=<x.y.z> bash scripts/lightning-setup.sh --prepare  # downloads on CPU
#   VLLM_VERSION=<x.y.z> RATE=<usd/hour> RATE_SOURCE="Lightning L4 on-demand, <date>" bash scripts/lightning-setup.sh
#   bash scripts/lightning-setup.sh --rehearsal  # CPU-only dry run with the mock vLLM server
#
# Steps: check GPU, install uv, sync the project (with the dashboard extra), install
# a pinned vLLM into its own venv, resolve and download the model, write the pinned
# configs, validate fixtures, and run doctor with the server started. Nothing here
# runs a measured cohort; see docs/LIGHTNING_RUNBOOK.md for the session itself.
set -euo pipefail

usage() {
  echo "usage: VLLM_VERSION=<x.y.z> bash scripts/lightning-setup.sh [--prepare|--rehearsal]"
  echo "  --prepare: install dependencies and download the model on CPU; do not start a server"
  echo "  --rehearsal: exercise setup with the mock server; no GPU or vLLM install"
}
MODE=run
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  "") ;;
  --prepare) MODE=prepare ;;
  --rehearsal) MODE=rehearsal ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE="${ROLLOUT_SETUP_PROFILE:-lightning}"
case "$PROFILE" in
  lightning) DEFAULT_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct; SMOKE_CONFIG=configs/smoke.pinned.yaml ;;
  notebook) DEFAULT_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ; SMOKE_CONFIG=configs/notebook-smoke.pinned.yaml ;;
  *) echo "unknown setup profile: $PROFILE" >&2; exit 2 ;;
esac
RUN_CONFIG="configs/$PROFILE.pinned.yaml"
MODEL_REPO="${MODEL_REPO:-$DEFAULT_MODEL}"
if [[ $MODE != rehearsal ]]; then
  : "${VLLM_VERSION:?set VLLM_VERSION to an exact vLLM release (e.g. one at least two weeks old)}"
fi
# Separate release directories keep a new session from changing an old session's
# default executable path. Explicit overrides are still synchronized below.
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv-${VLLM_VERSION:-rehearsal}}"

step() { printf '\n== %s\n' "$*"; }

step "GPU"
if [[ $MODE != run ]]; then
  echo "$MODE: skipping GPU check"
elif ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  echo "nvidia-smi failed: this is not a usable GPU host. Stop here (and stop the Studio)." >&2
  exit 1
fi
echo "CPUs available: $(nproc)"

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "project dependencies"
uv sync --python 3.12 --extra dashboard --locked

step "vLLM and model"
if [[ $MODE == rehearsal ]]; then
  echo "rehearsal: using the mock vLLM server; no vLLM install or model download"
  REVISION=rehearsal
  VLLM_BIN=vllm
else
  if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VLLM_VENV"
  fi
  # Always synchronize the requested release, including when an overridden
  # environment already contains a different vLLM. uv reuses satisfied packages.
  uv pip install --python "$VLLM_VENV/bin/python" "vllm==$VLLM_VERSION"
  INSTALLED_VLLM_VERSION="$("$VLLM_VENV/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')"
  if [[ "$INSTALLED_VLLM_VERSION" != "$VLLM_VERSION" ]]; then
    echo "vLLM version mismatch: requested $VLLM_VERSION, installed $INSTALLED_VLLM_VERSION in $VLLM_VENV" >&2
    exit 1
  fi
  VLLM_BIN="$VLLM_VENV/bin/vllm"
  if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM executable not found at $VLLM_BIN; repair this environment before continuing" >&2
    exit 1
  fi
  echo "vLLM $INSTALLED_VLLM_VERSION: $VLLM_BIN"
  uv pip freeze --python "$VLLM_VENV/bin/python" >"$ROOT/vllm-packages.txt"
  if [[ $MODE == run ]]; then
    "$VLLM_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable: check the driver and vLLM wheel compatibility"; assert torch.cuda.get_device_capability(0) >= (7, 5), "vLLM requires compute capability 7.5+: choose T4, not P100"; print("GPU:", torch.cuda.get_device_name(0), "Torch CUDA:", torch.version.cuda)'
    if ! "$VLLM_BIN" serve --help >"$ROOT/vllm-serve-help.txt" 2>&1; then
      echo "vllm serve --help failed; see vllm-serve-help.txt" >&2
      exit 1
    fi
    echo "saved 'vllm serve --help' to vllm-serve-help.txt: check every flag in configs/local.yaml against it"
  fi
  REVISION="$(uv run --with huggingface_hub python -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$MODEL_REPO")"
  echo "model $MODEL_REPO @ $REVISION"
  uv run --with huggingface_hub python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2])' "$MODEL_REPO" "$REVISION"
fi

if [[ $MODE == prepare ]]; then
  echo "CPU preparation complete. Switch to the GPU, restore the same VLLM_VERSION, MODEL_REPO and VLLM_VENV overrides, and rerun without --prepare."
  exit 0
fi

step "executor"
EXECUTOR=local
if [[ $PROFILE != notebook ]] && docker info >/dev/null 2>&1; then
  echo "docker daemon available: building the tool image and using the docker executor"
  docker build -q -t rollout-cost-tools:py312 -f docker/tool-runner.Dockerfile docker
  EXECUTOR=docker
else
  echo "using local tools; CPU allocation is detected below (no container isolation)"
fi

step "pinned configs"
PIN_ARGS=(--profile "$PROFILE" --model-repo "$MODEL_REPO" --revision "$REVISION" --vllm "$VLLM_BIN" --executor "$EXECUTOR")
[[ -n "${RATE:-}" ]] && PIN_ARGS+=(--rate "$RATE")
[[ -n "${RATE_SOURCE:-}" ]] && PIN_ARGS+=(--rate-source "$RATE_SOURCE")
[[ $MODE == rehearsal ]] && PIN_ARGS+=(--rehearsal)
uv run rollout-cost pin "${PIN_ARGS[@]}"

step "fixtures and doctor"
uv run rollout-cost validate-fixtures --config "$RUN_CONFIG"
uv run rollout-cost doctor --config "$RUN_CONFIG" --start-server

if [[ $PROFILE == notebook ]]; then
cat <<EOF

Notebook setup complete. Next (docs/NOTEBOOK_RUNBOOK.md):
  uv run rollout-cost run --config $SMOKE_CONFIG --split dev
EOF
else
cat <<EOF

Setup complete. Next (docs/LIGHTNING_RUNBOOK.md):
  bash scripts/observability-native.sh start             # optional live view
  uv run rollout-cost run --config $SMOKE_CONFIG --split dev
EOF
fi
