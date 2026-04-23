import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Standard masked language modeling objective:
    #
    #   L_MLM(x; theta)
    #     = -(1 / |M|) * sum_{t in M} log p_theta(x_t | x_\M)
    #
    # where M is the set of token positions that were corrupted by the
    # dynamic masking collator (80% replaced by [MASK], 10% replaced by a
    # random token, 10% left unchanged). In this module, `batch["labels"]`
    # already marks non-masked positions (and padding / special tokens)
    # with -100, so M is exactly the set of non-masked labels.
    #
    # For a minibatch, this becomes:
    #
    #   L_batch
    #     = -(
    #         sum_i sum_{t in M_i} log p_theta(x_{i,t} | x_{i,\M_i})
    #       ) / (
    #         sum_i |M_i|
    #       )
    #
    # We compute this manually instead of relying on HF's internal
    # `model(..., labels=...)` loss path.
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).logits

        # MLM has no causal shift: logits at position t predict the token
        # at position t directly.
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Only masked (supervised) positions should contribute to the
        # numerator and denominator. Unmasked / padding / special-token
        # positions were already converted to -100 in the collator.
        supervision_mask = batch["labels"].ne(-100)

        # `gather` requires valid class ids everywhere, so masked positions
        # are temporarily filled with 0 and zeroed out immediately afterward.
        safe_labels = batch["labels"].clone()
        safe_labels[~supervision_mask] = 0

        # token_log_probs[i, t] = log p_theta(x_{i,t} | x_{i,\M_i})
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        # Negative log-likelihood on valid MLM targets only.
        token_nll = -token_log_probs * supervision_mask

        supervised_token_count = supervision_mask.sum().clamp_min(1)
        loss = token_nll.sum() / supervised_token_count

    return loss
