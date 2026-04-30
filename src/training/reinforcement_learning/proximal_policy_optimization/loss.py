"""PPO clipped surrogate loss (critic-free, sequence-level advantage).

Equation (length-normalised form, length_normalize=True):
    L_PPO
        = - (1 / |y_i|) * sum_t mask_{i,t} * min(
              r_{i,t} * A_i,
              clip(r_{i,t}, 1 - eps, 1 + eps) * A_i
          )

Equation (raw-sum form, length_normalize=False):
    L_PPO
        = - sum_t mask_{i,t} * min(
              r_{i,t} * A_i,
              clip(r_{i,t}, 1 - eps, 1 + eps) * A_i
          )

with the per-token importance ratio
    r_{i,t} = exp(log pi_theta(y_{i,t}) - log pi_{theta_old}(y_{i,t}))
and the sequence-level advantage `A_i` (e.g. `R_i - mean_j(R_j)` from the
batch-mean baseline) broadcast across every completion token.

Why this shape?
- The token-level ratio with sequence-level advantage is the standard
  critic-free PPO loss used by GRPO / RLOO / TRL's PPOTrainer when no value
  head is trained. It lets a single trajectory-level reward signal drive a
  per-token policy update.
- The `min(...)` of the unclipped and clipped surrogates is what makes PPO
  stable across multiple gradient steps on the same rollout batch: when the
  policy drifts away from the rollout policy (|r - 1| > eps), the gradient
  is clipped to zero in the direction that would push it further away, so
  re-using rollouts is safe.
- With `num_ppo_epochs == 1` (i.e. only one gradient step per rollout) the
  ratio is identically 1 on the very first inner pass, the clip is inactive,
  and the loss collapses to REINFORCE-with-baseline.

Length normalisation matches `reinforce/loss.py`: dividing by the number of
supervised completion tokens prevents long rollouts from dominating purely on
token count and keeps the loss magnitude on a stable scale across batches.
"""

import torch


def forward_loss(
    policy_logps,
    old_logps,
    mask,
    advantages,
    clip_eps,
    length_normalize=False,
):
    # policy_logps: per-token log-probs from the CURRENT policy (gradient-carrying),
    #               shape [B, T-1].
    # old_logps:    per-token log-probs from the rollout-time policy (detached),
    #               shape [B, T-1].
    # mask:         shape [B, T-1] with 1.0 on completion tokens that should
    #               contribute to the gradient and 0.0 on prompt/padding.
    # advantages:   sequence-level advantages, shape [B], detached.
    # clip_eps:     PPO clipping range epsilon (e.g. 0.2).
    advantages = advantages.detach().unsqueeze(1)

    # Multiplying by `mask` here makes log_ratio (and therefore ratio) exactly
    # zero / one on padded positions, so they contribute nothing to either the
    # loss or the diagnostics regardless of what the model produced there.
    log_ratio = (policy_logps - old_logps) * mask
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token_loss = -torch.min(surr1, surr2) * mask

    if length_normalize:
        # clamp_min(1.0) protects against the degenerate case where a sample
        # has zero supervised tokens (prompt + EOS already filled max_length).
        seq_loss = per_token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = seq_loss.mean()
    else:
        loss = per_token_loss.sum(dim=1).mean()

    denom = mask.sum().clamp_min(1.0)
    clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float() * mask
    clip_frac = clipped.sum() / denom

    # k3 estimator of KL(pi_theta || pi_theta_old): unbiased, always >= 0,
    # cheaper to compute than the full KL and what most modern PPO codebases
    # use for the early-stop / logging signal.
    approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denom

    ratio_mean = (ratio * mask).sum() / denom

    return loss, ratio_mean, clip_frac, approx_kl


def reduce_kl_to_loss(loss, kl_value, kl_coeff):
    if kl_coeff is None or kl_coeff <= 0.0 or kl_value is None:
        return loss
    return loss + kl_coeff * kl_value
