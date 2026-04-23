import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Pairwise sigmoid contrastive loss:
    #
    # Replaces the softmax-based InfoNCE with a pairwise sigmoid, treating
    # each entry in the N x N similarity matrix independently:
    #
    #   L = -(1/N) * sum_{i,j} log sigma(z_ij * (t * sim(I_i, T_j) + b))
    #
    # where z_ij = +1 for matching pairs and -1 for non-matching,
    # t = exp(logit_scale) and b = logit_bias are learned parameters.
    #
    # Unlike softmax-based contrastive loss, no global normalization is
    # needed across the batch, which makes the loss more scalable to very
    # large batch sizes across many devices.
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        image_features = model.get_image_features(pixel_values=batch["pixel_values"])
        text_features = model.get_text_features(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logit_scale = model.logit_scale.exp()
        logit_bias = model.logit_bias

        logits = logit_scale * image_features @ text_features.t() + logit_bias

        n = logits.size(0)
        labels = 2 * torch.eye(n, device=logits.device) - 1

        loss = -F.logsigmoid(labels * logits).sum() / n

    return loss
