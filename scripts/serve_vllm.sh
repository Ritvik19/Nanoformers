#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server with the WeightSyncExtension wired in,
# so the training script can hot-swap weights via POST /collective_rpc.
#
# GPU placement is left to the caller. Example:
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh
#
# Override defaults via env vars:
#
#   MODEL=Qwen/Qwen3-0.6B PORT=8000 GPU_MEMORY_UTILIZATION=0.9 \
#       bash scripts/serve_vllm.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
WORKER_EXTENSION_CLS=${WORKER_EXTENSION_CLS:-src.training.reinforcement_learning.weight_sync.WeightSyncExtension}

# LoRA serving: set ENABLE_LORA=1 to allow the vLLM server to accept LoRA
# adapters loaded dynamically via the load_lora_adapter RPC. When disabled
# (default) the server works in standard full-weight mode — no behaviour change.
ENABLE_LORA=${ENABLE_LORA:-0}
MAX_LORAS=${MAX_LORAS:-1}
MAX_LORA_RANK=${MAX_LORA_RANK:-128}
MAX_CPU_LORAS=${MAX_CPU_LORAS:-2}

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# vLLM hides admin endpoints (/collective_rpc, /reset_prefix_cache, /sleep, ...)
# behind this flag. We need /collective_rpc for the weight-sync handshake and
# /reset_prefix_cache to invalidate stale KV after each reload.
export VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-1}

LORA_ARGS=""
if [ "${ENABLE_LORA}" = "1" ]; then
    LORA_ARGS="--enable-lora --max-loras ${MAX_LORAS} --max-lora-rank ${MAX_LORA_RANK} --max-cpu-loras ${MAX_CPU_LORAS}"
    # Required for the /v1/load_lora_adapter and /v1/unload_lora_adapter
    # endpoints to be registered. Without this the adapter can only be
    # specified at server startup via --lora-modules.
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
fi

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --worker-extension-cls "$WORKER_EXTENSION_CLS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    $LORA_ARGS
