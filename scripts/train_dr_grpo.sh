#!/usr/bin/env bash
# Launch the Dr. GRPO training loop. Reuses the GRPO trainer with toggles set
# to drop GRPO's two biases:
#   - std_normalize=false  -> no `(R_i - mean) / std`, just centered reward
#   - loss_aggregation=token -> token-level loss (no per-seq length normalisation)
#
# Requires a vLLM server already running (see scripts/serve_vllm.sh) at the
# URL specified in the YAML config.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/train_dr_grpo.sh
#
# Override the config file via env var:
#
#   CONFIG=configs/dr_grpo_qwen_math.yaml bash scripts/train_dr_grpo.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-configs/dr_grpo_qwen_math.yaml}

exec python -m src.cli.train_grpo --config "$CONFIG"
