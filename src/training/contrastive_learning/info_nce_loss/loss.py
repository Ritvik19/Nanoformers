import torch
import torch.nn.functional as F

from src.training.contrastive_learning.embed import encode


def forward_loss(model, batch, temperature=0.07):
    # InfoNCE loss (van den Oord et al., 2018 / SimCLR):
    #
    #   L_i = -log  exp(sim(z_i, z_i^+) / tau)
    #              --------------------------------
    #              sum_{j=1}^{2N} 1[j != i] exp(sim(z_i, z_j) / tau)
    #
    # Given a batch of N positive pairs, both views are encoded and every
    # other embedding in the 2N set serves as a negative.  The loss is
    # symmetric: each side of the pair takes a turn as the anchor.
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        anchor_emb = encode(
            model, batch["anchor_input_ids"], batch["anchor_attention_mask"]
        )
        positive_emb = encode(
            model, batch["positive_input_ids"], batch["positive_attention_mask"]
        )

        # [2N, D] — stack both views so every embedding is compared to
        # every other embedding in one similarity matrix.
        embeddings = torch.cat([anchor_emb, positive_emb], dim=0)
        n = anchor_emb.size(0)

        sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

        # Mask out self-similarity on the diagonal.
        mask = torch.eye(2 * n, device=sim_matrix.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # For anchor_i its positive is at index i + N, and vice-versa.
        targets = torch.cat(
            [torch.arange(n, 2 * n), torch.arange(0, n)], dim=0
        ).to(sim_matrix.device)

        loss = F.cross_entropy(sim_matrix, targets)

    return loss
