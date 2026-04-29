import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint
from src.training.common.config import load_config
from src.training.common.io import (
    load_token_classification_model,
    load_tokenizer,
)
from src.training.common.metrics import masked_accuracy, mean
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
from src.training.supervised_learning.token_classification.collator import collate_fn
from src.training.supervised_learning.token_classification.dataset import (
    TokenClassificationDataset,
    build_label_mappings,
    encode_labels,
    tokenize_and_align_labels,
)
from src.training.supervised_learning.token_classification.loss import forward_loss


def load_tokenizer_only(args):
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    print("Tokenizer loaded...")
    return tokenizer


def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    tokens_column = "tokens"
    labels_column = "labels"

    if tokens_column not in raw_dataset.column_names:
        raise ValueError(
            f"Dataset must contain a '{tokens_column}' column. Found: {raw_dataset.column_names}"
        )
    if labels_column not in raw_dataset.column_names:
        raise ValueError(
            f"Dataset must contain a '{labels_column}' column. Found: {raw_dataset.column_names}"
        )

    label_to_id, id_to_label = build_label_mappings(raw_dataset, labels_column)
    print(f"Using tokens column: {tokens_column}")
    print(f"Using labels column: {labels_column}")
    print(f"Resolved labels: {id_to_label}")

    print("Encoding labels...")
    labeled_dataset = raw_dataset.map(
        lambda example: encode_labels(example, labels_column, label_to_id),
        num_proc=256,
    )

    print("Tokenizing dataset...")
    tokenized_dataset = labeled_dataset.map(
        lambda example: tokenize_and_align_labels(
            example,
            tokenizer,
            tokens_column,
            args["max_length"],
            label_all_tokens=args.get("label_all_tokens", False),
        ),
        remove_columns=labeled_dataset.column_names,
        num_proc=256,
    )

    print("Tokenized dataset:")
    print(tokenized_dataset)

    if len(tokenized_dataset) < 2:
        raise ValueError("Token classification requires at least 2 examples.")

    print("Splitting dataset...")
    test_size = min(compute_test_size(len(tokenized_dataset)), len(tokenized_dataset) - 1)
    split = tokenized_dataset.train_test_split(test_size=test_size)

    train_dataset = TokenClassificationDataset(split["train"])
    eval_dataset = TokenClassificationDataset(split["test"])
    print(train_dataset)
    print(eval_dataset)

    print("Preparing dataloaders...")
    train_loader, eval_loader = build_dataloaders(
        train_dataset,
        eval_dataset,
        args["batch_size"],
        lambda batch: collate_fn(batch, tokenizer),
    )
    print("Dataset loaded and prepared...")
    return train_loader, eval_loader, label_to_id, id_to_label


def load_model(args, id_to_label):
    print("Loading model...")
    model = load_token_classification_model(
        args["model_path"],
        args["device"],
        num_labels=len(id_to_label),
    )
    model.config.label2id = {label: index for index, label in id_to_label.items()}
    model.config.id2label = id_to_label
    print("Model loaded...")
    return model


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
            "batch_size": args["batch_size"],
            "gradient_accumulation_steps": args["gradient_accumulation_steps"],
            "num_epochs": args["num_epochs"],
            "learning_rate": args["learning_rate"],
            "max_length": args["max_length"],
            "label_all_tokens": args.get("label_all_tokens", False),
        },
    )

    device = args["device"]
    global_step = 0
    ga = args["gradient_accumulation_steps"]
    # Accumulate raw preds/labels so token-accuracy is computed over the
    # concatenated effective batch instead of averaging per-micro-batch
    # accuracies (which would weight ragged sequences incorrectly when
    # masked_accuracy ignores -100 padding).
    accum = {"loss": 0.0, "predictions": [], "labels": [], "count": 0}
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss, logits = forward_loss(model, batch)
            scaled_loss = loss / ga
            scaler.scale(scaled_loss).backward()

            accum["loss"] += loss.item()
            accum["predictions"].extend(logits.argmax(dim=-1).detach().cpu().tolist())
            accum["labels"].extend(batch["labels"].detach().cpu().tolist())
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
                batch_accuracy = masked_accuracy(
                    accum["predictions"], accum["labels"]
                )

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/token_accuracy": batch_accuracy,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {
                        "loss": avg_loss,
                        "token_acc": batch_accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                accum = {"loss": 0.0, "predictions": [], "labels": [], "count": 0}

        model.eval()
        eval_losses = []
        eval_predictions = []
        eval_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                eval_loss, logits = forward_loss(model, batch)
                eval_losses.append(eval_loss.item())
                eval_predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                eval_labels.extend(batch["labels"].cpu().tolist())

        avg_eval_loss = mean(eval_losses)
        eval_accuracy = masked_accuracy(eval_predictions, eval_labels)
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/token_accuracy": eval_accuracy,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f} - Eval Token Accuracy: {eval_accuracy:.4f}"
        )

        save_checkpoint(model, tokenizer, args["output_dir"], epoch + 1)
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    tokenizer = load_tokenizer_only(args)
    train_loader, eval_loader, _, id_to_label = load_and_prepare_dataset(
        args,
        tokenizer,
    )
    model = load_model(args, id_to_label)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, scheduler)


if __name__ == "__main__":
    main()
