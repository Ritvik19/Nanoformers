"""vLLM worker extension that exposes weight-sync collective RPCs.

The vLLM serve script wires this class in via `--worker-extension-cls`, which
makes the method available on every vLLM worker. The training process can
trigger reloads by POSTing to the server's `/collective_rpc` endpoint.

Full-FT path (default):
    POST /collective_rpc {"method": "load_weights_from_path", "args": [path]}

LoRA / QLoRA path (requires ENABLE_LORA=1 when serving):
    POST /v1/load_lora_adapter {"lora_name": "...", "lora_path": "..."}
    POST /v1/unload_lora_adapter {"lora_name": "..."}

LoRA syncs use vLLM's native /v1/load_lora_adapter endpoint instead of
collective_rpc because the serving engine keeps its own adapter registry
separate from the GPU worker. Only the native endpoint updates both, avoiding
404 responses on completion requests after a collective_rpc-only load.
"""


class WeightSyncExtension:
    # Repoints the underlying ModelConfig at a fresh checkpoint directory and
    # asks the GPU model runner to reload weights in place. The default model
    # loader picks up `model_config.model` when it walks the checkpoint dir for
    # `*.safetensors` shards, so mutating the path redirects the reload.
    def load_weights_from_path(self, path: str) -> bool:
        self.model_runner.model_config.model = path
        self.model_runner.reload_weights()
        return True
