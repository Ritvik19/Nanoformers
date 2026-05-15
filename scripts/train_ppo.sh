#!/usr/bin/env bash
# Launch the PPO training loop. Requires a vLLM server already running
# (see scripts/serve_vllm.sh) at the URL specified in the YAML config.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/train_ppo.sh
#
# Override the config file via env var:
#
#   CONFIG=configs/ppo_qwen_math.yaml bash scripts/train_ppo.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-configs/lora/ppo_qwen_math_lora.yaml}

exec python -m src.cli.train_ppo --config "$CONFIG"
