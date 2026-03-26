import os


def save_checkpoint(model, tokenizer, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path
