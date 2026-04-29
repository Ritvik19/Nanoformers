"""Per-token log-probability and KL utilities shared across RL algorithms.

The shifted-logit gather pattern mirrors the existing DPO loss in
`src/training/supervised_learning/direct_preference_optimization/loss.py`.
"""

import torch
import torch.nn.functional as F


def _per_token_log_probs(logits, input_ids, attention_mask):
    # Causal LM alignment: position t in `logits` predicts token t+1.
    # We drop the last logit and the first input/attention column so that the
    # gathered log-probs are aligned with the *next* token at every position.
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()

    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp * mask, mask


def compute_per_token_log_probs(model, input_ids, attention_mask):
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
    return _per_token_log_probs(logits, input_ids, attention_mask)


def compute_per_token_log_probs_from_ref(ref_model, input_ids, attention_mask):
    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available(),
            dtype=torch.bfloat16,
        ):
            logits = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits
    return _per_token_log_probs(logits, input_ids, attention_mask)


def compute_kl_penalty(policy_logps, ref_logps, mask):
    # First-order approximation:
    #   KL = (1/T) * sum_t mask_t * [log pi_theta(y_t) - log pi_ref(y_t)]
    # averaged across the effective tokens of the batch.
    diff = (policy_logps - ref_logps) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
