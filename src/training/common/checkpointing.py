import os

import torch


def save_checkpoint(model, tokenizer, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path


def save_peft_checkpoint(model, tokenizer, output_dir, epoch, save_mode: str = "adapter"):
    """Save a PEFT (LoRA) checkpoint.

    Args:
        model: a PeftModel (or a plain nn.Module if PEFT is disabled).
        tokenizer: the tokenizer to save alongside.
        output_dir: root directory; epoch sub-directory is created automatically.
        epoch: epoch number, used as the sub-directory name.
        save_mode: one of
            "adapter"  – save only the adapter weights (default; smallest on disk).
            "merged"   – merge adapter into base and save a full HF model (largest).
            "both"     – save the adapter *and* a merged copy side-by-side.

    When the model is not a PeftModel (PEFT disabled) this falls back to the
    standard save_checkpoint path regardless of save_mode.
    """
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except ImportError:
        is_peft = False

    if not is_peft:
        return save_checkpoint(model, tokenizer, output_dir, epoch)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")

    if save_mode in ("adapter", "both"):
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

    if save_mode in ("merged", "both"):
        merged_path = os.path.join(output_dir, f"epoch_{epoch}_merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        if save_mode == "merged":
            return merged_path

    return checkpoint_path


def save_transient_checkpoint(model, tokenizer, output_dir, name="_vllm_sync"):
    """Used by RL pipelines for mid-training weight syncs into the inference engine.

    When the model is a PeftModel only the adapter is written (fast, ~30 MB).
    The full-FT path always overwrites the same directory so we don't accumulate
    one folder per step on disk; persistent epoch checkpoints still go through
    save_checkpoint / save_peft_checkpoint.
    """
    checkpoint_path = os.path.join(output_dir, name)
    os.makedirs(checkpoint_path, exist_ok=True)
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


def save_peft_dual_encoder_checkpoint(
    model, tokenizer, image_processor, output_dir, epoch, save_mode: str = "adapter"
):
    """Save a PEFT-wrapped DualEncoderModel checkpoint.

    The DualEncoderModel wraps two separate base encoders (text + image) each
    of which may have been independently wrapped with PEFT. We save each
    sub-model's adapter separately and then also save the projection heads
    (which are always full-precision trainable parameters).

    When PEFT is not active on either encoder, falls back to
    save_dual_encoder_checkpoint.
    """
    try:
        from peft import PeftModel
        text_is_peft = isinstance(model.text_encoder, PeftModel)
        image_is_peft = isinstance(model.image_encoder, PeftModel)
    except ImportError:
        text_is_peft = False
        image_is_peft = False

    if not text_is_peft and not image_is_peft:
        return save_dual_encoder_checkpoint(
            model, tokenizer, image_processor, output_dir, epoch
        )

    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")
    text_dir = os.path.join(checkpoint_path, "text_encoder")
    image_dir = os.path.join(checkpoint_path, "image_encoder")
    projection_path = os.path.join(checkpoint_path, "projection.pt")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    if save_mode in ("adapter", "both"):
        if text_is_peft:
            model.text_encoder.save_pretrained(text_dir)
        else:
            model.text_encoder.save_pretrained(text_dir)
        tokenizer.save_pretrained(text_dir)

        if image_is_peft:
            model.image_encoder.save_pretrained(image_dir)
        else:
            model.image_encoder.save_pretrained(image_dir)
        image_processor.save_pretrained(image_dir)

    if save_mode in ("merged", "both"):
        merged_text_dir = os.path.join(checkpoint_path, "text_encoder_merged")
        merged_image_dir = os.path.join(checkpoint_path, "image_encoder_merged")
        os.makedirs(merged_text_dir, exist_ok=True)
        os.makedirs(merged_image_dir, exist_ok=True)

        if text_is_peft:
            model.text_encoder.merge_and_unload().save_pretrained(merged_text_dir)
        else:
            model.text_encoder.save_pretrained(merged_text_dir)
        tokenizer.save_pretrained(merged_text_dir)

        if image_is_peft:
            model.image_encoder.merge_and_unload().save_pretrained(merged_image_dir)
        else:
            model.image_encoder.save_pretrained(merged_image_dir)
        image_processor.save_pretrained(merged_image_dir)

    projection_state = {
        "text_projection": model.text_projection.state_dict(),
        "image_projection": model.image_projection.state_dict(),
        "logit_scale": model.logit_scale.data,
    }
    if hasattr(model, "logit_bias"):
        projection_state["logit_bias"] = model.logit_bias.data
    torch.save(projection_state, projection_path)

    return checkpoint_path
