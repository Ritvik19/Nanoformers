import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Standard causal language modeling objective under teacher forcing:
    #
    #   L_CLM(x; theta)
    #     = -(1 / |M|) * sum_{t in M} log pi_theta(x_t | x_<t)
    #
    # where M is the set of target-token positions that should contribute to
    # the loss. In this module, `batch["labels"]` already marks padding
    # positions with -100, so M is exactly the set of non-masked labels after
    # the standard causal shift.
    #
    # For a minibatch, this becomes:
    #
    #   L_batch
    #     = -(
    #         sum_i sum_{t in M_i} log pi_theta(x_{i,t} | x_{i,<t})
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

        # Causal LM alignment:
        # logits at position t predict the token at position t + 1, so we
        # shift logits left and labels right before scoring.
        shifted_logits = logits[:, :-1, :]
        shifted_labels = batch["labels"][:, 1:]

        sequence_length = shifted_logits.size(1)
        shifted_labels = shifted_labels[:, :sequence_length]

        # Recover log pi_theta(token | prefix) terms from the equation above.
        log_probs = F.log_softmax(shifted_logits.float(), dim=-1)

        # Only non-masked labels should contribute to the numerator and
        # denominator. Padding positions were already converted to -100 in the
        # dataset.
        supervision_mask = shifted_labels.ne(-100)

        # `gather` requires valid class ids everywhere, so masked positions are
        # temporarily filled with 0 and zeroed out immediately afterward.
        safe_labels = shifted_labels.clone()
        safe_labels[~supervision_mask] = 0

        # token_log_probs[i, t] = log pi_theta(x_{i,t} | x_{i,<t})
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        # Negative log-likelihood on valid CLM targets only.
        token_nll = -token_log_probs * supervision_mask

        supervised_token_count = supervision_mask.sum().clamp_min(1)
        loss = token_nll.sum() / supervised_token_count

    return loss
