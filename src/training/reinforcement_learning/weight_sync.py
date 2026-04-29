"""vLLM worker extension that exposes a `load_weights_from_path` collective RPC.

The vLLM serve script wires this class in via `--worker-extension-cls`, which
makes `load_weights_from_path` a method of every vLLM worker. The training
process can then trigger an in-place weight reload by POSTing to the server's
`/collective_rpc` endpoint with `{"method": "load_weights_from_path", ...}`.
"""


class WeightSyncExtension:
    # Repoints the underlying ModelConfig at a fresh checkpoint directory and
    # asks the GPU model runner to reload weights in place. The default model
    # loader picks up `model_config.model` when it walks the checkpoint dir for
    # `*.safetensors` shards (see DefaultModelLoader.get_all_weights), so simply
    # mutating the path is enough to redirect the reload at the new checkpoint.
    def load_weights_from_path(self, path: str) -> bool:
        self.model_runner.model_config.model = path
        self.model_runner.reload_weights()
        return True
