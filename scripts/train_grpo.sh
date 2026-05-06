#!/usr/bin/env bash
# Launch the GRPO training loop. Requires a vLLM server already running
# (see scripts/serve_vllm.sh) at the URL specified in the YAML config.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/train_grpo.sh
#
# Override the config file via env var:
#
#   CONFIG=configs/grpo_qwen_math.yaml bash scripts/train_grpo.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-configs/grpo_qwen_math.yaml}

exec python -m src.cli.train_grpo --config "$CONFIG"
