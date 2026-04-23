import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint
from src.training.common.config import load_config
from src.training.common.io import (
    load_sequence_to_sequence_model,
    load_tokenizer,
)
from src.training.common.metrics import mean, perplexity_from_loss
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
from src.training.self_supervised_learning.span_corruption.collator import (
    collate_fn,
)
from src.training.self_supervised_learning.span_corruption.dataset import (
    SpanCorruptionDataset,
    group_texts,
    tokenize_function,
)
from src.training.self_supervised_learning.span_corruption.loss import (
    forward_loss,
)


def load_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    model = load_sequence_to_sequence_model(
        args["model_path"],
        args["device"],
        load_weights=args["load_weights"],
    )
    print("Model and tokenizer loaded...")
    return model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=256,
    )
    # Span corruption only needs raw input_ids at this stage; any per-token
    # column emitted by the tokenizer (e.g. attention_mask) would get out
    # of sync with the new block lengths produced by group_texts.
    extra_cols = [c for c in tokenized_dataset.column_names if c != "input_ids"]
    if extra_cols:
        tokenized_dataset = tokenized_dataset.remove_columns(extra_cols)

    print("Tokenized dataset:")
    print(tokenized_dataset)

    print("Splitting dataset...")
    split = tokenized_dataset.train_test_split(
        test_size=compute_test_size(len(tokenized_dataset))
    )

    print("Grouping texts into blocks...")
    max_length = args["max_length"]
    train_dataset = SpanCorruptionDataset(
        split["train"].map(
            lambda batch: group_texts(batch, block_size=max_length),
            batched=True,
            remove_columns=split["train"].column_names,
            num_proc=256,
        )
    )
    eval_dataset = SpanCorruptionDataset(
        split["test"].map(
            lambda batch: group_texts(batch, block_size=max_length),
            batched=True,
            remove_columns=split["test"].column_names,
            num_proc=256,
        )
    )
    print(train_dataset)
    print(eval_dataset)

    print("Preparing dataloaders...")
    train_loader, eval_loader = build_dataloaders(
        train_dataset,
        eval_dataset,
        args["batch_size"],
        lambda batch: collate_fn(
            batch,
            tokenizer,
            noise_density=args["noise_density"],
            mean_span_length=args["mean_span_length"],
        ),
    )
    print("Dataset loaded and prepared...")
    return train_loader, eval_loader


def prepare_optimizer_scaler_and_scheduler(args, model, train_loader):
    print("Preparing optimizer, scaler, and scheduler...")
    optimizer = build_optimizer(model, args["learning_rate"])
    scaler = build_grad_scaler()
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, scheduler


def train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, scheduler):
    print("Starting training...")
    wandb.init(
        project=args["wandb_project"],
        name=args["wandb_run_name"],
        config={
            "model_name": args["model_path"],
            "dataset_path": args["dataset_path"],
            "load_weights": args["load_weights"],
            "batch_size": args["batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "num_epochs": args["num_epochs"],
            "learning_rate": args["learning_rate"],
            "max_length": args["max_length"],
            "noise_density": args["noise_density"],
            "mean_span_length": args["mean_span_length"],
        },
    )

    device = args["device"]
    global_step = 0
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss = forward_loss(model, batch)
            scaled_loss = loss / args["gradient_accumulation_steps"]
            scaler.scale(scaled_loss).backward()

            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}
                )

        model.eval()
        eval_losses = []
        for batch in eval_loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                eval_losses.append(forward_loss(model, batch).item())

        avg_eval_loss = mean(eval_losses)
        perplexity = perplexity_from_loss(avg_eval_loss)
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/perplexity": perplexity,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f} - Perplexity: {perplexity:.2f}"
        )

        save_checkpoint(model, tokenizer, args["output_dir"], epoch + 1)
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    model, tokenizer = load_model_and_tokenizer(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, scheduler)


if __name__ == "__main__":
    main()
