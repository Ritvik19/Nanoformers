"""Thin client glue for talking to a standalone vLLM OpenAI-compatible server.

The training process uses the vanilla `openai.OpenAI` client for rollouts and a
small `httpx` wrapper for vLLM's admin endpoints (`/health`, `/collective_rpc`,
`/reset_prefix_cache`).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI


def build_openai_client(base_url, api_key="EMPTY"):
    return OpenAI(api_key=api_key, base_url=base_url)


def wait_for_server(admin_url, timeout_s=600, poll_interval_s=2.0):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{admin_url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"vLLM server at {admin_url} did not become healthy within {timeout_s}s "
        f"(last error: {last_error})"
    )


def _one_completion(
    client, model, prompt, max_new_tokens, temperature, top_p, top_k
):
    # `prompt` here is the chat template string already produced by
    # tokenizer.apply_chat_template(..., add_generation_prompt=True), so we
    # send it through the completions endpoint to avoid double templating.
    # top_k isn't part of the OpenAI spec, so it's smuggled in via extra_body
    # which vLLM forwards into SamplingParams. Set top_k=-1 to disable.
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k

    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body or None,
    )
    return resp.choices[0].text


def generate_rollouts(
    client,
    model,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
    top_k=None,
    n=1,
    max_workers=None,
):
    # Returns list[str] when n == 1 and list[list[str]] otherwise. The
    # multi-completion shape is here so PPO/GRPO can reuse this helper later.
    if max_workers is None:
        max_workers = max(1, len(prompts) * n)

    tasks = []
    for prompt in prompts:
        tasks.append([(prompt,) for _ in range(n)])

    flat = [(p_idx, sample_idx) for p_idx, samples in enumerate(tasks) for sample_idx in range(len(samples))]

    completions = [[None] * n for _ in prompts]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _one_completion,
                client,
                model,
                prompts[p_idx],
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ): (p_idx, sample_idx)
            for (p_idx, sample_idx) in flat
        }
        for future in futures:
            p_idx, sample_idx = futures[future]
            completions[p_idx][sample_idx] = future.result()

    if n == 1:
        return [row[0] for row in completions]
    return completions


def reload_weights(admin_url, checkpoint_path, timeout_s=600):
    resp = httpx.post(
        f"{admin_url}/collective_rpc",
        json={
            "method": "load_weights_from_path",
            "args": [checkpoint_path],
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else None


def reload_lora_adapter(admin_url, adapter_path, lora_name, timeout_s=120):
    """Tell the vLLM server to hot-swap the LoRA adapter.

    Uses vLLM's /v1/load_lora_adapter endpoint which registers the adapter
    with both the GPU worker *and* the API serving engine's model registry.
    Using collective_rpc → model_runner.add_lora() only touches the worker and
    leaves the serving layer unaware, causing 404s on completion requests.

    An unload is attempted first so that repeated syncs during training always
    replace the slot cleanly (the same path is overwritten each sync step).
    Requires the vLLM server to have been started with --enable-lora and
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.

    Args:
        admin_url:    e.g. "http://localhost:8000"
        adapter_path: path to the saved PeftModel adapter directory
                      (the one that contains adapter_model.safetensors).
                      Will be resolved to an absolute path.
        lora_name:    an arbitrary string identifier for the adapter;
                      rollout calls should pass this as the `model` argument.
        timeout_s:    request timeout.
    """
    adapter_path = os.path.abspath(adapter_path)

    # Unload the previous version if it exists so the reload is always clean.
    # Ignore 404 (first-time load) and any other non-fatal errors.
    try:
        httpx.post(
            f"{admin_url}/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
            timeout=30.0,
        )
    except Exception:
        pass

    resp = httpx.post(
        f"{admin_url}/v1/load_lora_adapter",
        json={"lora_name": lora_name, "lora_path": adapter_path},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    # vLLM returns a plain-text success message, not JSON.
    return resp.text if resp.content else None


def reset_prefix_cache(admin_url, timeout_s=60):
    resp = httpx.post(f"{admin_url}/reset_prefix_cache", timeout=timeout_s)
    resp.raise_for_status()
