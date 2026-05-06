#!/usr/bin/env bash
# Launch the DAPO training loop. Reuses the GRPO trainer with two toggles
# flipped from canonical GRPO:
#   - clip_high > clip_low ("Clip-Higher" — gives upside ratios more headroom)
#   - loss_aggregation=token (no per-seq length normalisation)
#
# Requires a vLLM server already running (see scripts/serve_vllm.sh) at the
# URL specified in the YAML config.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/train_dapo.sh
#
# Override the config file via env var:
#
#   CONFIG=configs/dapo_qwen_math.yaml bash scripts/train_dapo.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-configs/dapo_qwen_math.yaml}

exec python -m src.cli.train_grpo --config "$CONFIG"
