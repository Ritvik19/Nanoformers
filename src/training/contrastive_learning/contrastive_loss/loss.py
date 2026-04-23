import torch
import torch.nn.functional as F

from src.training.contrastive_learning.embed import encode


def forward_loss(model, batch, margin=1.0):
    # Pairwise contrastive loss (Hadsell et al., 2006):
    #
    #   L = y * D(a, b)^2  +  (1 - y) * max(0, margin - D(a, b))^2
    #
    # where y = 1 for similar pairs (pull together) and y = 0 for
    # dissimilar pairs (push apart beyond the margin).
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        anchor_emb = encode(model, batch["anchor_input_ids"], batch["anchor_attention_mask"])
        other_emb = encode(model, batch["other_input_ids"], batch["other_attention_mask"])

        distances = F.pairwise_distance(anchor_emb, other_emb)
        labels = batch["labels"].float()

        loss = labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2)
        loss = loss.mean()

    return loss
