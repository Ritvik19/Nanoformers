import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Standard instruction fine-tuning objective for one prompt/response pair
    # (x, y) under teacher forcing:
    #
    #   L_IFT(x, y; theta)
    #     = -(1 / |y|) * sum_{t=1}^{|y|} log pi_theta(y_t | x, y_<t)
    #
    # In this codebase, `batch["labels"]` already encodes the set of supervised
    # positions M by placing:
    # - the assistant response tokens at their target ids
    # - prompt tokens and padding tokens at -100
    #
    # So the minibatch loss becomes:
    #
    #   L_batch
    #     = -(
    #         sum_i sum_{t in M_i} log pi_theta(y_{i,t} | x_i, y_{i,<t})
    #       ) / (
    #         sum_i |M_i|
    #       )
    #
    # where M_i is the set of assistant-token positions that should contribute
    # to the loss for example i.
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).logits

        # Causal LM alignment:
        # logits at position t predict the token at position t + 1.
        # HF `model(..., labels=...)` shifts internally, but because we are
        # deriving the loss manually we must perform the shift ourselves.
        shifted_logits = logits[:, :-1, :]
        shifted_labels = batch["labels"][:, 1:]

        sequence_length = shifted_logits.size(1)
        shifted_labels = shifted_labels[:, :sequence_length]

        # Convert logits into log-probabilities so we can directly recover the
        # log pi_theta(y_t | x, y_<t) terms from the equation above.
        log_probs = F.log_softmax(shifted_logits.float(), dim=-1)

        # M_i from the equation is implemented via the `-100` mask:
        # only assistant completion tokens are supervised; prompt and padding
        # positions are excluded from both the numerator and denominator.
        supervision_mask = shifted_labels.ne(-100)

        # `gather` needs valid token ids everywhere, so we temporarily replace
        # masked positions with 0. Those entries are multiplied by 0
        # immediately afterward and therefore do not affect the loss.
        safe_labels = shifted_labels.clone()
        safe_labels[~supervision_mask] = 0

        # Per-token log-probabilities from the model:
        #   token_log_probs[i, t] = log pi_theta(y_{i,t} | x_i, y_{i,<t})
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        # Negative log-likelihood on the supervised positions only:
        #   -log pi_theta(...)
        token_nll = -token_log_probs * supervision_mask

        # Batch reduction matching the equation above:
        # sum all supervised token losses, then divide by the number of
        # supervised tokens. The clamp prevents a divide-by-zero if a rare
        # fully masked batch appears after truncation.
        supervised_token_count = supervision_mask.sum().clamp_min(1)
        loss = token_nll.sum() / supervised_token_count

    return loss
