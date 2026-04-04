import math

import torch
import torch.nn as nn


class DualEncoderModel(nn.Module):
    """Wraps independent text and image encoders with learned projections
    into a shared embedding space.

    Exposes the same API as CLIPModel / SiglipModel so the existing
    loss functions work without modification:
        .get_text_features(input_ids, attention_mask)
        .get_image_features(pixel_values)
        .logit_scale          (always present)
        .logit_bias           (present when use_logit_bias=True)
    """

    def __init__(
        self,
        text_encoder,
        image_encoder,
        text_hidden_size,
        image_hidden_size,
        projection_dim,
        use_logit_bias=False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)
        self.image_projection = nn.Linear(image_hidden_size, projection_dim, bias=False)

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07))
        )
        if use_logit_bias:
            self.logit_bias = nn.Parameter(torch.zeros(1))

    def get_text_features(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.text_projection(pooled)

    def get_image_features(self, pixel_values):
        out = self.image_encoder(pixel_values=pixel_values)
        pooled = out.last_hidden_state[:, 0]
        return self.image_projection(pooled)
