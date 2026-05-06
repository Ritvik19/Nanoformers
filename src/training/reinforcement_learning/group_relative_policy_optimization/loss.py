"""GRPO clipped surrogate loss with DAPO / Dr. GRPO / GSPO toggles.

Equation (token-level importance ratio, the standard PPO/GRPO/DAPO/Dr.GRPO form):
    r_{i,t} = exp(log pi_theta(y_{i,t}) - log pi_{theta_old}(y_{i,t}))
    L_TOK
        = - aggregate_t mask_{i,t} * min(
              r_{i,t} * A_i,
              clip(r_{i,t}, 1 - eps_low, 1 + eps_high) * A_i
          )

Equation (sequence-level importance ratio, GSPO):
    r_hat_i = exp( (1 / |y_i|) * sum_t mask_{i,t} * (log pi_theta(y_{i,t}) - log pi_{theta_old}(y_{i,t})) )
            = geometric mean of per-token ratios over the completion span
    L_SEQ
        = - mean_i min(
              r_hat_i * A_i,
              clip(r_hat_i, 1 - eps_low, 1 + eps_high) * A_i
          )

The advantage `A_i` is the group-relative advantage computed in the pipeline
over G samples per prompt (group-mean baseline, optionally divided by the
group std). This loss is agnostic to how `A_i` was produced — it just needs
one scalar per sequence in the (B*G,) flattened batch.

Toggle summary:
- `importance_ratio_level`:
    "token"    -> per-token ratio + per-token clip (PPO / GRPO / DAPO / Dr. GRPO).
    "sequence" -> length-normalised per-sequence ratio + per-sequence clip (GSPO).
- `loss_aggregation` (only meaningful when `importance_ratio_level=="token"`):
    "sequence" -> per-seq sum / |y_i|, then mean over batch (GRPO).
    "token"    -> sum over all tokens / sum(mask) (DAPO / Dr. GRPO).
  When `importance_ratio_level=="sequence"` the surrogate is already one
  scalar per sequence so this knob is ignored (loss is mean over sequences).
- `clip_low`, `clip_high`: asymmetric clip bounds. `clip_low == clip_high`
  recovers the symmetric PPO clip; DAPO uses a larger `clip_high` ("Clip-Higher").
  Per-token ratios drift fast (typical ~0.2); per-sequence ratios drift much
  more slowly (GSPO paper uses ~3e-4) — retune when switching the ratio level.
"""

import torch


def forward_loss(
    policy_logps,
    old_logps,
    mask,
    advantages,
    clip_low,
    clip_high,
    loss_aggregation,
    importance_ratio_level,
):
    # policy_logps: per-token log-probs from the CURRENT policy (gradient-carrying),
    #               shape [B, T-1].
    # old_logps:    per-token log-probs from the rollout-time policy (detached),
    #               shape [B, T-1].
    # mask:         shape [B, T-1] with 1.0 on completion tokens that should
    #               contribute to the gradient and 0.0 on prompt/padding.
    # advantages:   sequence-level advantages, shape [B], detached.
    # Multiplying by `mask` here makes log_ratio (and ratio) exactly zero / one
    # on padded positions, so they contribute nothing to either the loss or
    # the diagnostics regardless of what the model produced there.
    log_ratio_per_tok = (policy_logps - old_logps) * mask
    seq_lens = mask.sum(dim=1).clamp_min(1.0)
    denom = mask.sum().clamp_min(1.0)

    if importance_ratio_level == "token":
        adv = advantages.detach().unsqueeze(1)
        ratio = torch.exp(log_ratio_per_tok)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
        per_token_loss = -torch.min(surr1, surr2) * mask

        if loss_aggregation == "sequence":
            loss = (per_token_loss.sum(dim=1) / seq_lens).mean()
        elif loss_aggregation == "token":
            loss = per_token_loss.sum() / denom
        else:
            raise ValueError(
                f"unknown loss_aggregation={loss_aggregation!r}; "
                "expected 'sequence' or 'token'"
            )

        clipped = (
            (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
        ).float() * mask
        clip_frac = clipped.sum() / denom
        ratio_mean = (ratio * mask).sum() / denom

    elif importance_ratio_level == "sequence":
        # GSPO: one importance ratio per sequence, computed as the geometric
        # mean of per-token ratios. The clip is then applied to that scalar,
        # so clipping decisions are made per sequence rather than per token.
        # The gradient still flows through every per-token log-prob via the
        # sum inside seq_log_ratio.
        adv = advantages.detach()
        seq_log_ratio = log_ratio_per_tok.sum(dim=1) / seq_lens
        seq_ratio = torch.exp(seq_log_ratio)

        surr1 = seq_ratio * adv
        surr2 = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
        seq_loss = -torch.min(surr1, surr2)
        loss = seq_loss.mean()

        clipped = (
            (seq_ratio < 1.0 - clip_low) | (seq_ratio > 1.0 + clip_high)
        ).float()
        clip_frac = clipped.mean()
        ratio_mean = seq_ratio.mean()

    else:
        raise ValueError(
            f"unknown importance_ratio_level={importance_ratio_level!r}; "
            "expected 'token' or 'sequence'"
        )

    # k3 estimator of KL(pi_theta || pi_theta_old): unbiased, always >= 0.
    # Stays at token level even in GSPO mode because it's a drift diagnostic
    # measured per-token and matches what the PPO module logs.
    token_ratio = torch.exp(log_ratio_per_tok)
    approx_kl = (((token_ratio - 1.0) - log_ratio_per_tok) * mask).sum() / denom

    return loss, ratio_mean, clip_frac, approx_kl


def reduce_kl_to_loss(loss, kl_value, kl_coeff):
    if kl_coeff is None or kl_coeff <= 0.0 or kl_value is None:
        return loss
    return loss + kl_coeff * kl_value
