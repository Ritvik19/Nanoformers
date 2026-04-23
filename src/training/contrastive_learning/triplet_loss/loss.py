import torch
import torch.nn.functional as F

from src.training.contrastive_learning.embed import encode


def forward_loss(model, batch, margin=1.0):
    # Triplet margin loss (Schroff et al., 2015):
    #
    #   L = max(0, D(a, p) - D(a, n) + margin)
    #
    # Pulls anchor closer to the positive and pushes it away from the
    # negative by at least `margin` in Euclidean space.
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        anchor_emb = encode(
            model, batch["anchor_input_ids"], batch["anchor_attention_mask"]
        )
        positive_emb = encode(
            model, batch["positive_input_ids"], batch["positive_attention_mask"]
        )
        negative_emb = encode(
            model, batch["negative_input_ids"], batch["negative_attention_mask"]
        )

        pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb)

        loss = F.relu(pos_dist - neg_dist + margin).mean()

    return loss
