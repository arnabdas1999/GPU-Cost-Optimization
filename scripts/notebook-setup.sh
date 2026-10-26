#!/usr/bin/env bash
# Called from notebooks/gpu_smoke_test.ipynb on Colab or Kaggle.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROLLOUT_SETUP_PROFILE=notebook
export VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-Coder-3B-Instruct}"
# Keep large packages outside /kaggle/working and separate from the kernel.
export VLLM_VENV="${VLLM_VENV:-/tmp/rollout-cost-vllm-$VLLM_VERSION}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec bash "$ROOT/scripts/lightning-setup.sh" "$@"
