import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint, save_peft_checkpoint
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
from src.training.common.peft import (
    apply_peft_to_model,
    build_quantization_config,
    qlora_enabled,
)
from src.training.common.trainer import build_dataloaders
from src.training.common.utils import (
    compute_test_size,
    load_hf_dataset,
    move_batch_to_device,
)
from src.training.supervised_learning.sequence_to_sequence.collator import (
    collate_fn,
)
from src.training.supervised_learning.sequence_to_sequence.dataset import (
    SequenceToSequenceDataset,
    tokenize_function,
)
from src.training.supervised_learning.sequence_to_sequence.loss import (
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

    # If dataset has train split, we map on it. Otherwise map on the whole dictionary.
    # To handle 'raw_dataset.column_names' properly if it's a DatasetDict
    column_names = raw_dataset.column_names
    if isinstance(column_names, dict):
        column_names = next(iter(column_names.values()))

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda example: tokenize_function(
            example,
            tokenizer,
            args["max_length"],
        ),
        remove_columns=column_names,
        num_proc=256,
    )

    print("Tokenized dataset:")
    print(tokenized_dataset)

    # Some datasets are loaded as a single "train" split, some are not
    # So we'll get the dataset to split if it's not already split
    if hasattr(tokenized_dataset, "train_test_split"):
        if len(tokenized_dataset) < 2:
            raise ValueError("Sequence to sequence modeling requires at least 2 examples.")
            
        print("Splitting dataset...")
        test_size = min(compute_test_size(len(tokenized_dataset)), len(tokenized_dataset) - 1)
        split = tokenized_dataset.train_test_split(test_size=test_size)
    else:
        # Assumed datasetdict with "train" and "test" or "validation"
        if "test" in tokenized_dataset:
            split = {"train": tokenized_dataset["train"], "test": tokenized_dataset["test"]}
        elif "validation" in tokenized_dataset:
            split = {"train": tokenized_dataset["train"], "test": tokenized_dataset["validation"]}
        else:
            raise ValueError("Could not find validation/test split or split the dataset.")

    train_dataset = SequenceToSequenceDataset(split["train"])
    eval_dataset = SequenceToSequenceDataset(split["test"])
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
    return train_loader, eval_loader


def load_model(args):
    print("Loading model...")
    qcfg = build_quantization_config(args)
    model = load_sequence_to_sequence_model(
        args["model_path"],
        args["device"],
        load_weights=args.get("load_weights", True),
        quantization_config=qcfg,
    )
    model = apply_peft_to_model(model, args, task_type="SEQ_2_SEQ_LM")
    print("Model loaded...")
    return model


def prepare_optimizer_scaler_and_scheduler(args, model, train_loader):
    print("Preparing optimizer, scaler, and scheduler...")
    optimizer = build_optimizer(model, args["learning_rate"], paged=qlora_enabled(args))
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
        },
    )

    device = args["device"]
    global_step = 0
    ga = args["gradient_accumulation_steps"]
    # Perplexity is derived from the averaged loss (perplexity_from_loss is
    # convex, so perplexity-of-mean != mean-of-perplexity; the former is the
    # one that corresponds to the optimizer step we just took).
    accum = {"loss": 0.0, "count": 0}
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss, _ = forward_loss(model, batch)
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
                perplexity = perplexity_from_loss(avg_loss)

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/perplexity": perplexity,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {
                        "loss": avg_loss,
                        "ppl": perplexity,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                accum = {"loss": 0.0, "count": 0}

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                eval_loss, _ = forward_loss(model, batch)
                eval_losses.append(eval_loss.item())

        avg_eval_loss = mean(eval_losses)
        eval_perplexity = perplexity_from_loss(avg_eval_loss)
        
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/perplexity": eval_perplexity,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f} - Eval Perplexity: {eval_perplexity:.4f}"
        )

        save_peft_checkpoint(
            model, tokenizer, args["output_dir"], epoch + 1,
            save_mode=args.get("peft", {}).get("save_mode", "adapter"),
        )
        model.train()

    wandb.finish()
    print("Training finished.")


def main():
    args = load_config()
    tokenizer = load_tokenizer_only(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer)
    model = load_model(args)
    optimizer, scaler, scheduler = prepare_optimizer_scaler_and_scheduler(
        args, model, train_loader
    )
    train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, scheduler)


if __name__ == "__main__":
    main()
