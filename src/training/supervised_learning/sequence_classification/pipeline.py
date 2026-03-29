import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint
from src.training.common.config import load_config
from src.training.common.io import (
    load_sequence_classification_model,
    load_tokenizer,
)
from src.training.common.metrics import accuracy, mean
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
from src.training.supervised_learning.sequence_classification.collator import (
    collate_fn,
)
from src.training.supervised_learning.sequence_classification.dataset import (
    SequenceClassificationDataset,
    build_label_mappings,
    encode_label,
    tokenize_function,
)
from src.training.supervised_learning.sequence_classification.loss import (
    forward_loss,
)


def load_tokenizer_only(args):
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    print("Tokenizer loaded...")
    return tokenizer


def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    label_to_id, id_to_label = build_label_mappings(raw_dataset)
    print(f"Resolved labels: {label_to_id}")

    print("Encoding labels...")
    labeled_dataset = raw_dataset.map(
        lambda example: encode_label(example, label_to_id),
        num_proc=256,
    )

    print("Tokenizing dataset...")
    tokenized_dataset = labeled_dataset.map(
        lambda example: tokenize_function(
            example,
            tokenizer,
            args["max_length"],
        ),
        remove_columns=[
            column_name
            for column_name in labeled_dataset.column_names
            if column_name != "label"
        ],
        num_proc=256,
    )

    print("Tokenized dataset:")
    print(tokenized_dataset)

    if len(tokenized_dataset) < 2:
        raise ValueError("Sequence classification requires at least 2 examples.")

    print("Splitting dataset...")
    test_size = min(compute_test_size(len(tokenized_dataset)), len(tokenized_dataset) - 1)
    split = tokenized_dataset.train_test_split(test_size=test_size)

    train_dataset = SequenceClassificationDataset(split["train"])
    eval_dataset = SequenceClassificationDataset(split["test"])
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


def load_model(args, label_to_id, id_to_label):
    print("Loading model...")
    model = load_sequence_classification_model(
        args["model_path"],
        args["device"],
        num_labels=len(label_to_id),
        load_weights=args.get("load_weights", True),
    )
    model.config.label2id = {str(label): index for label, index in label_to_id.items()}
    model.config.id2label = id_to_label
    model.config.problem_type = "single_label_classification"
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
            "load_weights": args.get("load_weights", True),
        },
    )

    device = args["device"]
    global_step = 0
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss, logits = forward_loss(model, batch)
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

                batch_predictions = logits.argmax(dim=-1).detach().cpu().tolist()
                batch_labels = batch["labels"].detach().cpu().tolist()
                batch_accuracy = accuracy(batch_predictions, batch_labels)

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": batch_accuracy,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": batch_accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

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
        eval_accuracy = accuracy(eval_predictions, eval_labels)
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/accuracy": eval_accuracy,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f} - Eval Accuracy: {eval_accuracy:.4f}"
        )

        save_checkpoint(model, tokenizer, args["output_dir"], epoch + 1)
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    tokenizer = load_tokenizer_only(args)
    train_loader, eval_loader, label_to_id, id_to_label = load_and_prepare_dataset(
        args,
        tokenizer,
    )
    model = load_model(args, label_to_id, id_to_label)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, scheduler)


if __name__ == "__main__":
    main()
