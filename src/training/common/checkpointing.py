import os

import torch


def save_checkpoint(model, tokenizer, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path


def save_dual_encoder_checkpoint(model, tokenizer, image_processor, output_dir, epoch):
    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")

    text_dir = os.path.join(checkpoint_path, "text_encoder")
    image_dir = os.path.join(checkpoint_path, "image_encoder")
    projection_path = os.path.join(checkpoint_path, "projection.pt")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    model.text_encoder.save_pretrained(text_dir)
    tokenizer.save_pretrained(text_dir)

    model.image_encoder.save_pretrained(image_dir)
    image_processor.save_pretrained(image_dir)

    projection_state = {
        "text_projection": model.text_projection.state_dict(),
        "image_projection": model.image_projection.state_dict(),
        "logit_scale": model.logit_scale.data,
    }
    if hasattr(model, "logit_bias"):
        projection_state["logit_bias"] = model.logit_bias.data
    torch.save(projection_state, projection_path)

    return checkpoint_path
