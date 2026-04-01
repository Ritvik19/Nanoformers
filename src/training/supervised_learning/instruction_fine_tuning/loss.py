import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Standard instruction fine-tuning objective for one prompt/response pair
    # (x, y) under teacher forcing:
    #
    #   L_IFT(x, y; theta)
    #     = -(1 / |y|) * sum_{t=1}^{|y|} log pi_theta(y_t | x, y_<t)
    #
    # The `batch["labels"]` already encodes the set of supervised
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

        # M_i from the equation is implemented via `ignore_index=-100`:
        # only assistant completion tokens are supervised; prompt and padding
        # positions are excluded from both the numerator and denominator
        # (same mean as the manual log_softmax + gather formulation).
        if shifted_labels.eq(-100).all():
            # Match the old clamp_min(1) denominator: no supervised tokens → 0 loss.
            loss = shifted_logits.sum() * 0.0
        else:
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )

    return loss
