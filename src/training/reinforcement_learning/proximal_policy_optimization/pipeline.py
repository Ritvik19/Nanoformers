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
from src.training.reinforcement_learning.policy_utils import (
    compute_kl_penalty,
    compute_per_token_log_probs,
    compute_per_token_log_probs_from_ref,
)
from src.training.reinforcement_learning.proximal_policy_optimization.loss import (
    forward_loss,
    reduce_kl_to_loss,
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
    # build_scheduler computes total_steps as
    #   num_epochs * ceil(loader_len / gradient_accumulation_steps),
    # which matches one optimizer step per ga micro-batches. PPO takes
    # `num_ppo_epochs` optimizer steps per ga micro-batches, so we scale the
    # effective loader length by num_ppo_epochs to keep warmup/decay aligned
    # with the *true* number of optimizer steps. With num_ppo_epochs=1 this
    # collapses back to REINFORCE's schedule exactly.
    num_ppo_epochs = int(args.get("num_ppo_epochs", 1))
    scheduler = build_scheduler(
        args, optimizer, len(train_loader) * num_ppo_epochs
    )
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, scheduler


def _rollout_step(args, model, ref_model, tokenizer, batch, device, sampling):
    # No-grad rollout phase: produce completions, score them, tokenize, and
    # snapshot per-token log-probs from BOTH the current policy (`old_logps`,
    # used as the importance-ratio denominator inside PPO) and the frozen
    # reference (used for the KL-to-ref penalty if kl_coeff > 0). All tensors
    # returned here are detached and reused across every PPO inner epoch.
    temperature, top_p, top_k = sampling

    completions = generate_rollouts(
        client=sampling_state["client"],
        model=args["vllm_served_model_name"],
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
        "rewards": rewards,
        "old_logps": old_logps.detach(),
        "ref_logps": ref_logps,
        "effective_mask": effective_mask,
    }


def _gradient_step(args, model, cached, advantages_micro):
    # With-grad phase: re-run the policy forward on a cached micro-batch, build
    # the clipped surrogate against the frozen `old_logps`, and (optionally)
    # add the KL-to-reference penalty. The reference log-probs are already
    # cached so the ref model is only called once per rollout, not once per
    # PPO inner epoch.
    tensors = cached["tensors"]
    policy_logps, _ = compute_per_token_log_probs(
        model, tensors["input_ids"], tensors["attention_mask"]
    )

    loss, ratio_mean, clip_frac, approx_kl = forward_loss(
        policy_logps=policy_logps,
        old_logps=cached["old_logps"],
        mask=cached["effective_mask"],
        advantages=advantages_micro,
        clip_eps=float(args["clip_eps"]),
        length_normalize=bool(args.get("length_normalize", False)),
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


def _compute_advantages(rollout_cache, use_baseline):
    # Advantages are computed over the FULL rollout effective batch (= ga
    # micro-batches) so the baseline `mean(R)` is estimated from the largest
    # sample size we have available. This is strictly lower-variance than
    # REINFORCE's per-micro-batch baseline.
    all_rewards = torch.cat([c["rewards"] for c in rollout_cache], dim=0)
    if use_baseline:
        advantages = all_rewards - all_rewards.mean()
    else:
        advantages = all_rewards.clone()
    return all_rewards.detach(), advantages.detach()


def _eval_step(args, model, ref_model, tokenizer, batch, device, sampling):
    # Eval is a pure rollout: generate greedily, score against ground truth,
    # report the reward. We don't need importance-ratio / clipped-loss values
    # because no gradient is taken.
    cached = _rollout_step(args, model, ref_model, tokenizer, batch, device, sampling)
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
            "clip_eps": float(args["clip_eps"]),
            "num_ppo_epochs": int(args.get("num_ppo_epochs", 1)),
            "target_kl": (
                float(args["target_kl"]) if args.get("target_kl") is not None else None
            ),
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
    num_ppo_epochs = int(args.get("num_ppo_epochs", 1))
    use_baseline = bool(args.get("use_baseline", False))
    target_kl = (
        float(args["target_kl"]) if args.get("target_kl") is not None else None
    )

    # vLLM sync is keyed off "rollout effective batches" rather than optimizer
    # steps because PPO takes num_ppo_epochs optimizer steps per rollout. The
    # natural unit for staleness control is the rollout itself.
    sync_every = int(args.get("vllm_sync_every_rollouts", 1))

    global_step = 0
    rollout_batch_idx = 0
    model.train()
    rollout_cache = []

    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            cached = _rollout_step(
                args, model, ref_model, tokenizer, batch, device, train_sampling
            )
            rollout_cache.append(cached)

            if (step + 1) % ga != 0:
                continue

            rollout_batch_idx += 1
            all_rewards, advantages = _compute_advantages(rollout_cache, use_baseline)
            offsets = [0]
            for c in rollout_cache:
                offsets.append(offsets[-1] + c["rewards"].numel())

            for ppo_ep in range(num_ppo_epochs):
                accum = {
                    "loss": 0.0,
                    "ratio_mean": 0.0,
                    "clip_frac": 0.0,
                    "approx_kl": 0.0,
                    "kl": 0.0,
                    "count": 0,
                }
                for mi, micro in enumerate(rollout_cache):
                    advs_micro = advantages[offsets[mi] : offsets[mi + 1]]
                    loss, metrics = _gradient_step(args, model, micro, advs_micro)

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
                # optimizer step (they recompute the policy forward each PPO
                # inner epoch, so they genuinely change). `reward_mean`,
                # `reward_std`, `advantage_mean` are derived from the FROZEN
                # rollout cache and therefore identical across PPO epochs of
                # the same rollout batch -- log them only on ppo_ep == 0 so
                # wandb gets one reward data point per rollout instead of K.
                log_dict = {
                    "train/loss": avg_loss,
                    "train/kl": avg_kl,
                    "train/ratio_mean": avg_ratio,
                    "train/clip_frac": avg_clip_frac,
                    "train/approx_kl": avg_approx_kl,
                    "train/ppo_epoch": ppo_ep,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/rollout_batch": rollout_batch_idx,
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                }
                if ppo_ep == 0:
                    log_dict["train/reward_mean"] = all_rewards.mean().item()
                    log_dict["train/reward_std"] = (
                        all_rewards.std(unbiased=False).item()
                        if all_rewards.numel() > 1
                        else 0.0
                    )
                    log_dict["train/advantage_mean"] = advantages.mean().item()

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
                # remaining PPO inner epochs to avoid trust-region violations.
                # The next outer iteration will collect a fresh rollout batch
                # under the (slightly drifted) policy regardless.
                if target_kl is not None and avg_approx_kl > target_kl:
                    break

            rollout_cache = []
            del all_rewards, advantages

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
