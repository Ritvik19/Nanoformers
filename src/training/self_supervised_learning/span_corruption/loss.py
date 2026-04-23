import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Standard T5 span corruption objective under teacher forcing:
    #
    #   L_span(x; theta)
    #     = -(1 / |T|) * sum_{t in T} log p_theta(y_t | y_<t, x_tilde)
    #
    # where x_tilde is the corrupted input sequence (noise spans replaced by
    # sentinel tokens), y is the sentinel-interleaved target sequence
    # (sentinel_i followed by the tokens of the i-th noise span, ..., final
    # sentinel, </s>), and T is the set of target positions that should
    # contribute to the loss. In this module, `batch["labels"]` already
    # marks padding positions with -100, so T is exactly the set of
    # non-masked labels.
    #
    # For a minibatch, this becomes:
    #
    #   L_batch
    #     = -(
    #         sum_i sum_{t in T_i} log p_theta(y_{i,t} | y_{i,<t}, x_tilde_i)
    #       ) / (
    #         sum_i |T_i|
    #       )
    #
    # HF's T5 forward pass automatically shifts `labels` right to build
    # `decoder_input_ids`, so `logits[:, t, :]` directly predicts
    # `labels[:, t]` with no extra shift on our side. We compute this
    # manually instead of relying on HF's internal `model(..., labels=...)`
    # loss path, to keep the self-supervised training modules consistent.
    #
    # T5's activations (notably the relative-position-bias + attention
    # path) routinely exceed fp16's ~6.5e4 dynamic range and overflow to
    # inf, which then becomes NaN in the softmax/log_softmax. bfloat16
    # has the same exponent range as fp32, so we cast explicitly here to
    # keep the objective numerically stable.
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )
        logits = outputs.logits

        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Only valid target tokens contribute to the numerator and
        # denominator. Padding positions were converted to -100 in the
        # collator.
        supervision_mask = batch["labels"].ne(-100)

        # `gather` requires valid class ids everywhere, so masked positions
        # are temporarily filled with 0 and zeroed out immediately afterward.
        safe_labels = batch["labels"].clone()
        safe_labels[~supervision_mask] = 0

        # token_log_probs[i, t] = log p_theta(y_{i,t} | y_{i,<t}, x_tilde_i)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        token_nll = -token_log_probs * supervision_mask

        supervised_token_count = supervision_mask.sum().clamp_min(1)
        loss = token_nll.sum() / supervised_token_count

    return loss
