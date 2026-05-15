import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import (
    save_checkpoint,
    save_peft_checkpoint,
    save_transient_checkpoint,
)
from src.training.common.config import load_config
from src.training.common.io import (
    load_causal_lm_model,
    load_reference_model,
    load_tokenizer,
)
from src.training.common.metrics import mean
from src.training.common.optim import (
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
)
from src.training.common.peft import (
    apply_peft_to_model,
    build_quantization_config,
    peft_enabled,
    qlora_enabled,
    use_disable_adapter_as_ref,
)
from src.training.common.trainer import build_dataloaders
from src.training.common.utils import (
    compute_test_size,
    load_hf_dataset,
)
from src.training.reinforcement_learning.collator import collate_fn
from src.training.reinforcement_learning.dataset import RLPromptDataset
from src.training.reinforcement_learning.policy_utils import (
    compute_kl_penalty,
    compute_per_token_log_probs,
    compute_per_token_log_probs_from_ref,
    compute_per_token_log_probs_from_ref_via_adapter,
)
from src.training.reinforcement_learning.reinforce.loss import (
    forward_loss,
    reduce_kl_to_loss,
)
from src.training.reinforcement_learning.reward import compute_outcome_rewards
from src.training.reinforcement_learning.tokenization import prepare_training_batch
from src.training.reinforcement_learning.vllm_client import (
    build_openai_client,
    generate_rollouts,
    reload_lora_adapter,
    reload_weights,
    reset_prefix_cache,
    wait_for_server,
)


def load_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    qcfg = build_quantization_config(args)
    model = load_causal_lm_model(
        args["model_path"], args["device"], load_weights=True, quantization_config=qcfg
    )
    model = apply_peft_to_model(model, args, task_type="CAUSAL_LM")

    ref_model = None
    if float(args.get("kl_coeff", 0.0)) > 0.0:
        if use_disable_adapter_as_ref(args):
            print("PEFT enabled — KL ref will be computed via disable_adapter().")
        else:
            print("Loading reference model (kl_coeff > 0)...")
            ref_model = load_reference_model(args["model_path"], args["device"])

    print("Model and tokenizer loaded...")
    return model, ref_model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    print("Splitting dataset...")
    split = raw_dataset.train_test_split(
        test_size=compute_test_size(len(raw_dataset))
    )

    train_dataset = RLPromptDataset(split["train"], tokenizer)
    eval_dataset = RLPromptDataset(split["test"], tokenizer)
    print(train_dataset)
    print(eval_dataset)

    print("Preparing dataloaders...")
    train_loader, eval_loader = build_dataloaders(
        train_dataset,
        eval_dataset,
        args["batch_size"],
        collate_fn,
    )
    print("Dataset loaded and prepared...")
    return train_loader, eval_loader


def prepare_optimizer_scaler_and_scheduler(args, model, train_loader):
    print("Preparing optimizer, scaler, and scheduler...")
    optimizer = build_optimizer(model, args["learning_rate"], paged=qlora_enabled(args))
    scaler = build_grad_scaler()
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, scheduler


def _compute_step(args, model, ref_model, tokenizer, batch, device, sampling):
    # `sampling` is a (temperature, top_p, top_k) tuple so eval can do greedy
    # decoding by overriding just temperature/top_p without losing the top_k.
    temperature, top_p, top_k = sampling

    # When PEFT is enabled the LoRA adapter name is used as the served model so
    # vLLM routes the request to the correct adapter rather than the bare base.
    _vllm_model = (
        args.get("peft", {}).get("vllm_lora_name", "peft_adapter")
        if peft_enabled(args)
        else args["vllm_served_model_name"]
    )
    completions = generate_rollouts(
        client=sampling_state["client"],
        model=_vllm_model,
        prompts=batch["prompts"],
        max_new_tokens=args["max_new_tokens"],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1,
    )

    rewards_list = compute_outcome_rewards(completions, batch["answers"])
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

    tensors = prepare_training_batch(
        tokenizer=tokenizer,
        prompts=batch["prompts"],
        completions=completions,
        max_length=args["max_length"],
        device=device,
    )

    policy_logps, attn_mask = compute_per_token_log_probs(
        model, tensors["input_ids"], tensors["attention_mask"]
    )
    completion_mask_shift = tensors["completion_mask"][:, 1:]
    effective_mask = attn_mask * completion_mask_shift

    kl_value = None
    if ref_model is not None:
        ref_logps, _ = compute_per_token_log_probs_from_ref(
            ref_model, tensors["input_ids"], tensors["attention_mask"]
        )
        kl_value = compute_kl_penalty(policy_logps, ref_logps, effective_mask)
    elif peft_enabled(args) and float(args.get("kl_coeff", 0.0)) > 0.0:
        ref_logps, _ = compute_per_token_log_probs_from_ref_via_adapter(
            model, tensors["input_ids"], tensors["attention_mask"]
        )
        kl_value = compute_kl_penalty(policy_logps, ref_logps, effective_mask)

    loss, advantages, _ = forward_loss(
        policy_logps=policy_logps,
        mask=effective_mask,
        rewards=rewards,
        use_baseline=bool(args.get("use_baseline", False)),
        length_normalize=bool(args.get("length_normalize", False)),
    )
    loss = reduce_kl_to_loss(loss, kl_value, float(args.get("kl_coeff", 0.0)))

    metrics = {
        "loss": loss.detach().float().item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(unbiased=False).item() if rewards.numel() > 1 else 0.0,
        "advantage_mean": advantages.mean().item(),
        "kl": kl_value.detach().float().item() if kl_value is not None else 0.0,
        # Raw rewards (per-sample) so the train loop can compute true mean/std
        # over the full effective batch instead of averaging stds.
        "rewards": rewards.detach().cpu().tolist(),
    }
    return loss, metrics


# Module-level container so `_compute_step` can stay a pure helper while still
# sharing the long-lived OpenAI client across the loop.
sampling_state = {"client": None}


def train(
    args,
    model,
    ref_model,
    tokenizer,
    train_loader,
    eval_loader,
    optimizer,
    scaler,
    scheduler,
):
    print("Starting training...")
    wandb.init(
        project=args["wandb_project"],
        name=args["wandb_run_name"],
        config={
            "model_name": args["model_path"],
            "dataset_path": args["dataset_path"],
            "batch_size": args["batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "num_epochs": args["num_epochs"],
            "learning_rate": args["learning_rate"],
            "max_length": args["max_length"],
            "max_new_tokens": args["max_new_tokens"],
            "temperature": args["temperature"],
            "top_p": args["top_p"],
            "top_k": args.get("top_k"),
            "use_baseline": bool(args.get("use_baseline", False)),
            "length_normalize": bool(args.get("length_normalize", False)),
            "kl_coeff": float(args.get("kl_coeff", 0.0)),
            "vllm_sync_every_steps": int(args.get("vllm_sync_every_steps", 1)),
        },
    )

    print("Waiting for vLLM server...")
    wait_for_server(args["vllm_admin_url"])
    sampling_state["client"] = build_openai_client(
        base_url=args["vllm_base_url"],
        api_key=args.get("vllm_api_key", "EMPTY"),
    )

    # Push the initial adapter to vLLM before the first rollout. Without this
    # the server only knows the base model name and would return 404 on the
    # first generate_rollouts call when `vllm_lora_name` is used as the model.
    if peft_enabled(args):
        lora_name = args.get("peft", {}).get("vllm_lora_name", "peft_adapter")
        print(f"Uploading initial LoRA adapter '{lora_name}' to vLLM...")
        init_sync_path = save_transient_checkpoint(model, tokenizer, args["output_dir"])
        reload_lora_adapter(args["vllm_admin_url"], init_sync_path, lora_name)
        reset_prefix_cache(args["vllm_admin_url"])
        print("Initial LoRA adapter uploaded.")

    device = args["device"]
    top_k = args.get("top_k")
    train_sampling = (args["temperature"], args["top_p"], top_k)
    # Greedy eval: removes sampling noise so reported eval reward tracks true policy quality.
    eval_sampling = (0.0, 1.0, top_k)

    # On-policy REINFORCE wants vLLM rollouts to come from the *current* policy.
    # Every sync = save_pretrained(disk) + POST /collective_rpc (reload). Default 1
    # means "every optimizer step"; raise to amortise sync cost at the price of
    # off-policy drift. Set <= 0 to disable mid-epoch sync (epoch-only behaviour).
    sync_every = int(args.get("vllm_sync_every_steps", 1))

    global_step = 0
    model.train()
    ga = args["gradient_accumulation_steps"]
    # Per-optimizer-step accumulators. We log aggregates over the *effective*
    # batch (= ga micro-batches) instead of just the last micro-batch, so
    # logged scalars match the gradient that was actually taken.
    accum = {"loss": 0.0, "advantage_mean": 0.0, "kl": 0.0, "rewards": [], "count": 0}

    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            loss, metrics = _compute_step(
                args, model, ref_model, tokenizer, batch, device, train_sampling
            )

            scaled_loss = loss / ga
            scaler.scale(scaled_loss).backward()

            accum["loss"] += metrics["loss"]
            accum["advantage_mean"] += metrics["advantage_mean"]
            accum["kl"] += metrics["kl"]
            accum["rewards"].extend(metrics["rewards"])
            accum["count"] += 1

            if (step + 1) % ga == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                n = accum["count"]
                avg_loss = accum["loss"] / n
                avg_advantage = accum["advantage_mean"] / n
                avg_kl = accum["kl"] / n
                rewards_t = torch.tensor(accum["rewards"], dtype=torch.float32)
                reward_mean = rewards_t.mean().item()
                reward_std = (
                    rewards_t.std(unbiased=False).item()
                    if rewards_t.numel() > 1
                    else 0.0
                )

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/reward_mean": reward_mean,
                        "train/reward_std": reward_std,
                        "train/advantage_mean": avg_advantage,
                        "train/kl": avg_kl,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {
                        "loss": avg_loss,
                        "rew": reward_mean,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                accum = {
                    "loss": 0.0,
                    "advantage_mean": 0.0,
                    "kl": 0.0,
                    "rewards": [],
                    "count": 0,
                }

                if sync_every > 0 and global_step % sync_every == 0:
                    sync_path = save_transient_checkpoint(
                        model, tokenizer, args["output_dir"]
                    )
                    if peft_enabled(args):
                        lora_name = args.get("peft", {}).get("vllm_lora_name", "peft_adapter")
                        reload_lora_adapter(args["vllm_admin_url"], sync_path, lora_name)
                    else:
                        reload_weights(args["vllm_admin_url"], sync_path)
                    reset_prefix_cache(args["vllm_admin_url"])

        model.eval()
        eval_rewards = []
        eval_kls = []
        with torch.no_grad():
            for batch in eval_loader:
                _, eval_metrics = _compute_step(
                    args, model, ref_model, tokenizer, batch, device, eval_sampling
                )
                eval_rewards.append(eval_metrics["reward_mean"])
                eval_kls.append(eval_metrics["kl"])

        avg_eval_reward = mean(eval_rewards) if eval_rewards else 0.0
        avg_eval_kl = mean(eval_kls) if eval_kls else 0.0
        wandb.log(
            {
                "eval/reward_mean": avg_eval_reward,
                "eval/kl": avg_eval_kl,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval reward: {avg_eval_reward:.4f} - Eval KL: {avg_eval_kl:.4f}"
        )

        checkpoint_path = save_peft_checkpoint(
            model, tokenizer, args["output_dir"], epoch + 1,
            save_mode=args.get("peft", {}).get("save_mode", "adapter"),
        )

        # If mid-epoch sync is disabled (sync_every <= 0), vLLM is still on
        # last-epoch weights at this point, so push the freshly-saved persistent
        # checkpoint. With sync_every > 0 vLLM is already up to date from the
        # last in-loop sync, so skip the redundant reload.
        if sync_every <= 0:
            print(f"Syncing vLLM weights from {checkpoint_path}...")
            if peft_enabled(args):
                lora_name = args.get("peft", {}).get("vllm_lora_name", "peft_adapter")
                reload_lora_adapter(args["vllm_admin_url"], checkpoint_path, lora_name)
            else:
                reload_weights(args["vllm_admin_url"], checkpoint_path)
            reset_prefix_cache(args["vllm_admin_url"])
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    model, ref_model, tokenizer = load_model_and_tokenizer(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(
        args,
        model,
        ref_model,
        tokenizer,
        train_loader,
        eval_loader,
        optimizer,
        scaler,
        scheduler,
    )


if __name__ == "__main__":
    main()
