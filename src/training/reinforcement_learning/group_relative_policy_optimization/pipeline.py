import warnings

import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint, save_transient_checkpoint
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
from src.training.common.trainer import build_dataloaders
from src.training.common.utils import (
    compute_test_size,
    load_hf_dataset,
)
from src.training.reinforcement_learning.collator import collate_fn
from src.training.reinforcement_learning.dataset import RLPromptDataset
from src.training.reinforcement_learning.group_relative_policy_optimization.loss import (
    forward_loss,
    reduce_kl_to_loss,
)
from src.training.reinforcement_learning.policy_utils import (
    compute_kl_penalty,
    compute_per_token_log_probs,
    compute_per_token_log_probs_from_ref,
)
from src.training.reinforcement_learning.reward import compute_outcome_rewards
from src.training.reinforcement_learning.tokenization import prepare_training_batch
from src.training.reinforcement_learning.vllm_client import (
    build_openai_client,
    generate_rollouts,
    reload_weights,
    reset_prefix_cache,
    wait_for_server,
)


def load_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    model = load_causal_lm_model(args["model_path"], args["device"], load_weights=True)

    ref_model = None
    if float(args.get("kl_coeff", 0.0)) > 0.0:
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
    optimizer = build_optimizer(model, args["learning_rate"])
    scaler = build_grad_scaler()
    # Match PPO's schedule scaling: one optimizer step per ga micro-batches per
    # GRPO inner epoch, so total optimizer steps = num_epochs * ceil(loader_len/ga)
    # * num_grpo_epochs. Scaling the effective loader length by num_grpo_epochs
    # keeps warmup/decay aligned. With num_grpo_epochs=1 this collapses to the
    # REINFORCE schedule.
    num_grpo_epochs = int(args.get("num_grpo_epochs", 1))
    scheduler = build_scheduler(
        args, optimizer, len(train_loader) * num_grpo_epochs
    )
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, scheduler


def _compute_group_advantages(rewards_bg, std_normalize):
    # rewards_bg: shape [B, G]. Returns advantages of the same shape.
    #
    # Group baseline = mean across the G samples in the group, broadcast to
    # every member. With std_normalize=True this reproduces the canonical
    # GRPO advantage `(R_i - mean) / std`; with False it's Dr. GRPO's
    # centered reward `R_i - mean`.
    eps = 1e-8
    mean_g = rewards_bg.mean(dim=-1, keepdim=True)
    centered = rewards_bg - mean_g
    if std_normalize:
        std_g = rewards_bg.std(dim=-1, unbiased=False, keepdim=True)
        return centered / (std_g + eps)
    return centered


def _rollout_step(args, model, ref_model, tokenizer, batch, device, sampling, group_size):
    # No-grad rollout phase: sample G completions per prompt from vLLM, score
    # each (prompt, completion) pair, compute group-relative advantages over
    # the [B, G] reward tensor, then tokenize the FLATTENED B*G sequences and
    # snapshot per-token log-probs from the current policy (`old_logps`, used
    # as the importance-ratio denominator in the clipped surrogate) and the
    # frozen reference (used for the KL-to-ref penalty if kl_coeff > 0).
    temperature, top_p, top_k = sampling
    B = len(batch["prompts"])
    G = group_size

    completions_groups = generate_rollouts(
        client=sampling_state["client"],
        model=args["vllm_served_model_name"],
        prompts=batch["prompts"],
        max_new_tokens=args["max_new_tokens"],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=G,
    )

    # vLLM client returns list[str] when n==1, else list[list[str]]. We always
    # request n=G here, but G==1 collapses the shape so normalise both cases.
    if G == 1:
        completions_groups = [[c] for c in completions_groups]

    flat_prompts = [p for p in batch["prompts"] for _ in range(G)]
    flat_completions = [c for grp in completions_groups for c in grp]
    flat_answers = [a for a in batch["answers"] for _ in range(G)]

    rewards_flat = compute_outcome_rewards(flat_completions, flat_answers)
    rewards_bg = torch.tensor(
        rewards_flat, dtype=torch.float32, device=device
    ).view(B, G)

    if G > 1:
        advantages_bg = _compute_group_advantages(
            rewards_bg,
            std_normalize=bool(args.get("std_normalize", True)),
        )
    else:
        # G == 1 is only ever used by `_eval_step`, which never reads
        # `advantages` — populate with zeros to keep the return shape stable
        # (and to side-step the degenerate "group of one" case where the
        # mean baseline equals the sample itself, so the centered advantage
        # is identically zero anyway).
        advantages_bg = torch.zeros_like(rewards_bg)

    tensors = prepare_training_batch(
        tokenizer=tokenizer,
        prompts=flat_prompts,
        completions=flat_completions,
        max_length=args["max_length"],
        device=device,
    )

    with torch.no_grad():
        old_logps, attn_mask = compute_per_token_log_probs(
            model, tensors["input_ids"], tensors["attention_mask"]
        )
    completion_mask_shift = tensors["completion_mask"][:, 1:]
    effective_mask = attn_mask * completion_mask_shift

    ref_logps = None
    if ref_model is not None:
        ref_logps, _ = compute_per_token_log_probs_from_ref(
            ref_model, tensors["input_ids"], tensors["attention_mask"]
        )
        ref_logps = ref_logps.detach()

    return {
        "tensors": tensors,
        "rewards": rewards_bg.flatten().detach(),
        "advantages": advantages_bg.flatten().detach(),
        "old_logps": old_logps.detach(),
        "ref_logps": ref_logps,
        "effective_mask": effective_mask,
    }


def _gradient_step(args, model, cached):
    # With-grad phase: re-run the policy forward on a cached micro-batch,
    # build the clipped surrogate against the frozen `old_logps` and the
    # cached group-relative advantages, and (optionally) add the KL-to-
    # reference penalty. Reference log-probs are cached so the ref model is
    # only called once per rollout, not once per GRPO inner epoch.
    tensors = cached["tensors"]
    policy_logps, _ = compute_per_token_log_probs(
        model, tensors["input_ids"], tensors["attention_mask"]
    )

    loss, ratio_mean, clip_frac, approx_kl = forward_loss(
        policy_logps=policy_logps,
        old_logps=cached["old_logps"],
        mask=cached["effective_mask"],
        advantages=cached["advantages"],
        clip_low=float(args["clip_low"]),
        clip_high=float(args["clip_high"]),
        loss_aggregation=str(args.get("loss_aggregation", "sequence")),
        importance_ratio_level=str(args.get("importance_ratio_level", "token")),
    )

    kl_value = None
    if cached["ref_logps"] is not None:
        kl_value = compute_kl_penalty(
            policy_logps, cached["ref_logps"], cached["effective_mask"]
        )
    loss = reduce_kl_to_loss(loss, kl_value, float(args.get("kl_coeff", 0.0)))

    metrics = {
        "loss": loss.detach().float().item(),
        "ratio_mean": ratio_mean.detach().float().item(),
        "clip_frac": clip_frac.detach().float().item(),
        "approx_kl": approx_kl.detach().float().item(),
        "kl": kl_value.detach().float().item() if kl_value is not None else 0.0,
    }
    return loss, metrics


def _eval_step(args, model, ref_model, tokenizer, batch, device, sampling):
    # Eval is greedy single-sample (n=1) — group structure isn't needed because
    # we don't compute advantages, just the per-prompt reward.
    cached = _rollout_step(
        args, model, ref_model, tokenizer, batch, device, sampling, group_size=1
    )
    rewards = cached["rewards"]
    kl_value = 0.0
    if cached["ref_logps"] is not None:
        with torch.no_grad():
            policy_logps, _ = compute_per_token_log_probs(
                model,
                cached["tensors"]["input_ids"],
                cached["tensors"]["attention_mask"],
            )
            kl_value = compute_kl_penalty(
                policy_logps, cached["ref_logps"], cached["effective_mask"]
            ).item()
    return {
        "reward_mean": rewards.mean().item(),
        "kl": kl_value,
    }


# Module-level container so the rollout/gradient helpers can stay pure while
# still sharing the long-lived OpenAI client across the loop.
sampling_state = {"client": None}


def _validate_args(args):
    G = int(args.get("group_size", 1))
    if G < 2:
        raise ValueError(
            f"group_size must be >= 2 for group-relative advantages "
            f"(got group_size={G}); use the REINFORCE pipeline for G=1."
        )

    ratio_level = str(args.get("importance_ratio_level", "token"))
    if ratio_level not in ("token", "sequence"):
        raise ValueError(
            f"importance_ratio_level must be 'token' or 'sequence', "
            f"got {ratio_level!r}"
        )

    loss_agg = str(args.get("loss_aggregation", "sequence"))
    if loss_agg not in ("sequence", "token"):
        raise ValueError(
            f"loss_aggregation must be 'sequence' or 'token', got {loss_agg!r}"
        )

    if ratio_level == "sequence" and "loss_aggregation" in args:
        warnings.warn(
            "importance_ratio_level='sequence' (GSPO): the surrogate is one "
            "scalar per sequence, so loss_aggregation is ignored "
            "(loss = mean over sequences).",
            stacklevel=2,
        )


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
    _validate_args(args)
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
            "group_size": int(args["group_size"]),
            "importance_ratio_level": str(args.get("importance_ratio_level", "token")),
            "loss_aggregation": str(args.get("loss_aggregation", "sequence")),
            "clip_low": float(args["clip_low"]),
            "clip_high": float(args["clip_high"]),
            "std_normalize": bool(args.get("std_normalize", True)),
            "num_grpo_epochs": int(args.get("num_grpo_epochs", 1)),
            "target_kl": (
                float(args["target_kl"]) if args.get("target_kl") is not None else None
            ),
            "kl_coeff": float(args.get("kl_coeff", 0.0)),
            "vllm_sync_every_rollouts": int(args.get("vllm_sync_every_rollouts", 1)),
        },
    )

    print("Waiting for vLLM server...")
    wait_for_server(args["vllm_admin_url"])
    sampling_state["client"] = build_openai_client(
        base_url=args["vllm_base_url"],
        api_key=args.get("vllm_api_key", "EMPTY"),
    )

    device = args["device"]
    top_k = args.get("top_k")
    train_sampling = (args["temperature"], args["top_p"], top_k)
    eval_sampling = (0.0, 1.0, top_k)

    ga = args["gradient_accumulation_steps"]
    group_size = int(args["group_size"])
    num_grpo_epochs = int(args.get("num_grpo_epochs", 1))
    target_kl = (
        float(args["target_kl"]) if args.get("target_kl") is not None else None
    )

    # vLLM sync is keyed off "rollout effective batches" (= ga micro-batches)
    # rather than optimizer steps because each rollout drives num_grpo_epochs
    # optimizer steps. The natural unit for staleness control is the rollout.
    sync_every = int(args.get("vllm_sync_every_rollouts", 1))

    global_step = 0
    rollout_batch_idx = 0
    model.train()
    rollout_cache = []

    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            cached = _rollout_step(
                args,
                model,
                ref_model,
                tokenizer,
                batch,
                device,
                train_sampling,
                group_size,
            )
            rollout_cache.append(cached)

            if (step + 1) % ga != 0:
                continue

            rollout_batch_idx += 1
            all_rewards = torch.cat([c["rewards"] for c in rollout_cache], dim=0)
            all_advantages = torch.cat(
                [c["advantages"] for c in rollout_cache], dim=0
            )

            for grpo_ep in range(num_grpo_epochs):
                accum = {
                    "loss": 0.0,
                    "ratio_mean": 0.0,
                    "clip_frac": 0.0,
                    "approx_kl": 0.0,
                    "kl": 0.0,
                    "count": 0,
                }
                for micro in rollout_cache:
                    loss, metrics = _gradient_step(args, model, micro)

                    scaled_loss = loss / ga
                    scaler.scale(scaled_loss).backward()

                    accum["loss"] += metrics["loss"]
                    accum["ratio_mean"] += metrics["ratio_mean"]
                    accum["clip_frac"] += metrics["clip_frac"]
                    accum["approx_kl"] += metrics["approx_kl"]
                    accum["kl"] += metrics["kl"]
                    accum["count"] += 1

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                n = max(accum["count"], 1)
                avg_loss = accum["loss"] / n
                avg_ratio = accum["ratio_mean"] / n
                avg_clip_frac = accum["clip_frac"] / n
                avg_approx_kl = accum["approx_kl"] / n
                avg_kl = accum["kl"] / n

                # `loss / ratio_mean / clip_frac / approx_kl / kl` are per
                # optimizer step (they recompute the policy forward each GRPO
                # inner epoch, so they genuinely change). `reward_mean`,
                # `reward_std`, `advantage_mean` are derived from the FROZEN
                # rollout cache and therefore identical across GRPO epochs of
                # the same rollout batch -- log them only on grpo_ep == 0 so
                # wandb gets one reward data point per rollout instead of K.
                log_dict = {
                    "train/loss": avg_loss,
                    "train/kl": avg_kl,
                    "train/ratio_mean": avg_ratio,
                    "train/clip_frac": avg_clip_frac,
                    "train/approx_kl": avg_approx_kl,
                    "train/grpo_epoch": grpo_ep,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/rollout_batch": rollout_batch_idx,
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                }
                if grpo_ep == 0:
                    log_dict["train/reward_mean"] = all_rewards.mean().item()
                    log_dict["train/reward_std"] = (
                        all_rewards.std(unbiased=False).item()
                        if all_rewards.numel() > 1
                        else 0.0
                    )
                    log_dict["train/advantage_mean"] = all_advantages.mean().item()
                    log_dict["train/advantage_std"] = (
                        all_advantages.std(unbiased=False).item()
                        if all_advantages.numel() > 1
                        else 0.0
                    )

                wandb.log(log_dict, step=global_step)

                loop.set_postfix(
                    {
                        "loss": avg_loss,
                        "rew": all_rewards.mean().item(),
                        "ratio": avg_ratio,
                        "clip": avg_clip_frac,
                        "kl_old": avg_approx_kl,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                # Optional KL-based early stop: if the policy has drifted too
                # far from the rollout policy on this rollout batch, abort the
                # remaining GRPO inner epochs to avoid trust-region violations.
                # The next outer iteration will collect a fresh rollout batch
                # under the (slightly drifted) policy regardless.
                if target_kl is not None and avg_approx_kl > target_kl:
                    break

            rollout_cache = []
            del all_rewards, all_advantages

            if (
                sync_every > 0
                and rollout_batch_idx % sync_every == 0
            ):
                sync_path = save_transient_checkpoint(
                    model, tokenizer, args["output_dir"]
                )
                reload_weights(args["vllm_admin_url"], sync_path)
                reset_prefix_cache(args["vllm_admin_url"])

        model.eval()
        eval_rewards = []
        eval_kls = []
        with torch.no_grad():
            for batch in eval_loader:
                eval_metrics = _eval_step(
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

        checkpoint_path = save_checkpoint(model, tokenizer, args["output_dir"], epoch + 1)

        if sync_every <= 0:
            print(f"Syncing vLLM weights from {checkpoint_path}...")
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
