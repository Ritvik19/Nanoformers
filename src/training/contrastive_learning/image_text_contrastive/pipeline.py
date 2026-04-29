import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_dual_encoder_checkpoint
from src.training.common.config import load_config
from src.training.common.io import (
    load_image_text_contrastive_model,
    load_image_text_processor,
    load_image_text_tokenizer,
)
from src.training.common.metrics import mean
from src.training.common.optim import (
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
)
from src.training.common.trainer import build_dataloaders
from src.training.common.utils import (
    compute_test_size,
    load_hf_dataset,
    move_batch_to_device,
)
from src.training.contrastive_learning.image_text_contrastive.collator import (
    collate_fn,
)
from src.training.contrastive_learning.image_text_contrastive.dataset import (
    ImageTextContrastiveDataset,
)
from src.training.contrastive_learning.image_text_contrastive.loss import forward_loss


def load_tokenizer_and_image_processor(args):
    print("Loading tokenizer and image processor...")
    tokenizer = load_image_text_tokenizer(args["text_model_path"])
    image_processor = load_image_text_processor(args["image_model_path"])
    print("Tokenizer and image processor loaded...")
    return tokenizer, image_processor


def load_and_prepare_dataset(args, tokenizer, image_processor):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    if len(raw_dataset) < 2:
        raise ValueError("Image-text contrastive training requires at least 2 examples.")

    print("Splitting dataset...")
    test_size = min(compute_test_size(len(raw_dataset)), len(raw_dataset) - 1)
    split = raw_dataset.train_test_split(test_size=test_size)

    train_dataset = ImageTextContrastiveDataset(split["train"])
    eval_dataset = ImageTextContrastiveDataset(split["test"])
    print(train_dataset)
    print(eval_dataset)

    max_length = args["max_length"]
    print("Preparing dataloaders...")
    train_loader, eval_loader = build_dataloaders(
        train_dataset,
        eval_dataset,
        args["batch_size"],
        lambda batch: collate_fn(batch, tokenizer, image_processor, max_length),
    )
    print("Dataset loaded and prepared...")
    return train_loader, eval_loader


def load_model(args):
    print("Loading model...")
    model = load_image_text_contrastive_model(
        args["text_model_path"],
        args["image_model_path"],
        args["projection_dim"],
        args["device"],
    )
    print("Model loaded...")
    return model


def prepare_optimizer_scaler_and_scheduler(args, model, train_loader):
    print("Preparing optimizer, scaler, and scheduler...")
    optimizer = build_optimizer(model, args["learning_rate"])
    scaler = build_grad_scaler()
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, scheduler


def train(args, model, tokenizer, image_processor, train_loader, eval_loader, optimizer, scaler, scheduler):
    print("Starting training...")
    wandb.init(
        project=args["wandb_project"],
        name=args["wandb_run_name"],
        config={
            "text_model_name": args["text_model_path"],
            "image_model_name": args["image_model_path"],
            "projection_dim": args["projection_dim"],
            "dataset_path": args["dataset_path"],
            "batch_size": args["batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "num_epochs": args["num_epochs"],
            "learning_rate": args["learning_rate"],
            "max_length": args["max_length"],
        },
    )

    device = args["device"]
    global_step = 0
    ga = args["gradient_accumulation_steps"]
    accum = {"loss": 0.0, "count": 0}
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss = forward_loss(model, batch)
            scaled_loss = loss / ga
            scaler.scale(scaled_loss).backward()

            accum["loss"] += loss.item()
            accum["count"] += 1

            if (step + 1) % ga == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                avg_loss = accum["loss"] / accum["count"]

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {"loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
                )

                accum = {"loss": 0.0, "count": 0}

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                eval_loss = forward_loss(model, batch)
                eval_losses.append(eval_loss.item())

        avg_eval_loss = mean(eval_losses)
        wandb.log(
            {"eval/loss": avg_eval_loss, "eval/epoch": epoch + 1},
            step=global_step,
        )
        print(f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f}")

        save_dual_encoder_checkpoint(model, tokenizer, image_processor, args["output_dir"], epoch + 1)
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    tokenizer, image_processor = load_tokenizer_and_image_processor(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer, image_processor)
    model = load_model(args)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(args, model, tokenizer, image_processor, train_loader, eval_loader, optimizer, scaler, scheduler)


if __name__ == "__main__":
    main()
