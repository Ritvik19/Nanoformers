import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    # Symmetric InfoNCE over image-text cosine similarities scaled by a
    # learned temperature t = exp(logit_scale):
    #
    #   L_i2t = CE(t * I @ T^T,  arange(N))
    #   L_t2i = CE(t * T @ I^T,  arange(N))
    #   L     = (L_i2t + L_t2i) / 2
    #
    # where I and T are L2-normalised image and text embeddings.
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
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(image_features.size(0), device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

    return loss
