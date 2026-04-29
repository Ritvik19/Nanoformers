"""REINFORCE policy-gradient loss.

Equation (length-normalised form, length_normalize=True):
    L_REINFORCE
        = -(R_i - b) * (1 / |y_i|) * sum_t mask_{i,t} * log pi_theta(y_{i,t} | x_i, y_{i,<t})

Equation (raw-sum form, length_normalize=False):
    L_REINFORCE
        = -(R_i - b) * sum_t mask_{i,t} * log pi_theta(y_{i,t} | x_i, y_{i,<t})

with `b = mean_i(R_i)` when `use_baseline` is enabled and `b = 0` otherwise.

The baseline subtraction does not change the expected gradient (it has zero
expectation under the policy) but it cuts variance dramatically when rewards
have a non-zero mean.

Length normalisation divides by the number of supervised completion tokens
|y_i| (i.e. mask.sum(dim=1)). The unbiased policy-gradient estimator uses the
raw sum, but in practice the mean form is overwhelmingly preferred: it
prevents long rollouts from dominating the gradient purely on token count and
keeps the loss magnitude on a stable scale across batches.
"""

import torch


def forward_loss(policy_logps, mask, rewards, use_baseline, length_normalize=False):
    # policy_logps: per-token log-probs from the policy (already shifted),
    #               shape [B, T-1], gradient-carrying.
    # mask:         shape [B, T-1] with 1.0 on completion tokens that should
    #               contribute to the gradient and 0.0 on prompt/padding.
    # rewards:      scalar reward per sequence, shape [B].
    seq_logprobs = (policy_logps * mask).sum(dim=1)
    if length_normalize:
        # clamp_min(1.0) protects against the degenerate case where a sample
        # has zero supervised tokens (e.g. the prompt + EOS already filled
        # max_length so the completion was truncated to nothing).
        seq_logprobs = seq_logprobs / mask.sum(dim=1).clamp_min(1.0)

    if use_baseline:
        baseline = rewards.mean()
        advantages = rewards - baseline
    else:
        advantages = rewards

    advantages = advantages.detach()

    loss = -(advantages * seq_logprobs).mean()
    return loss, advantages, seq_logprobs


def reduce_kl_to_loss(loss, kl_value, kl_coeff):
    if kl_coeff is None or kl_coeff <= 0.0 or kl_value is None:
        return loss
    return loss + kl_coeff * kl_value
