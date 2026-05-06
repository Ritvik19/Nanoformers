#!/usr/bin/env bash
# Launch the GSPO training loop. Reuses the GRPO trainer with the importance
# ratio computed once per sequence (geometric mean of per-token ratios) and
# clipping applied to that scalar — set via importance_ratio_level=sequence
# in the config. The clip range is much tighter (~3e-4) than for token-level
# methods because per-sequence ratios drift far more slowly.
#
# Requires a vLLM server already running (see scripts/serve_vllm.sh) at the
# URL specified in the YAML config.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/train_gspo.sh
#
# Override the config file via env var:
#
#   CONFIG=configs/gspo_qwen_math.yaml bash scripts/train_gspo.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-configs/gspo_qwen_math.yaml}

exec python -m src.cli.train_grpo --config "$CONFIG"
