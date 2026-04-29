import torch
import wandb
from tqdm.auto import tqdm

from src.training.common.checkpointing import save_checkpoint
from src.training.common.config import load_config
from src.training.common.io import (
    load_question_answering_model,
    load_tokenizer,
)
from src.training.common.metrics import mean, qa_exact_match
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
from src.training.supervised_learning.question_answering.collator import collate_fn
from src.training.supervised_learning.question_answering.dataset import (
    QuestionAnsweringDataset,
    tokenize_and_align_answers,
)
from src.training.supervised_learning.question_answering.loss import forward_loss


def load_tokenizer_only(args):
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args["model_path"])
    print("Tokenizer loaded...")
    return tokenizer


def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")
    raw_dataset = load_hf_dataset(args["dataset_path"])

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda example: tokenize_and_align_answers(
            example,
            tokenizer,
            args["max_length"],
        ),
        remove_columns=raw_dataset.column_names,
        num_proc=256,
    )

    print("Tokenized dataset:")
    print(tokenized_dataset)

    if len(tokenized_dataset) < 2:
        raise ValueError("Question answering requires at least 2 examples.")

    print("Splitting dataset...")
    test_size = min(compute_test_size(len(tokenized_dataset)), len(tokenized_dataset) - 1)
    split = tokenized_dataset.train_test_split(test_size=test_size)

    train_dataset = QuestionAnsweringDataset(split["train"])
    eval_dataset = QuestionAnsweringDataset(split["test"])
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
    model = load_question_answering_model(
        args["model_path"],
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
    # Accumulate raw start/end preds & labels so exact-match is computed over
    # the concatenated effective batch (mean-of-batch-EMs is biased on uneven
    # tail batches and obscures rare-correct cases).
    accum = {
        "loss": 0.0,
        "start_preds": [],
        "end_preds": [],
        "start_labels": [],
        "end_labels": [],
        "count": 0,
    }
    model.train()
    for epoch in range(args["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}")
        for step, batch in enumerate(loop):
            batch = move_batch_to_device(batch, device)
            loss, logits = forward_loss(model, batch)
            start_logits, end_logits = logits

            scaled_loss = loss / ga
            scaler.scale(scaled_loss).backward()

            accum["loss"] += loss.item()
            accum["start_preds"].extend(
                start_logits.argmax(dim=-1).detach().cpu().tolist()
            )
            accum["end_preds"].extend(
                end_logits.argmax(dim=-1).detach().cpu().tolist()
            )
            accum["start_labels"].extend(
                batch["start_positions"].detach().cpu().tolist()
            )
            accum["end_labels"].extend(
                batch["end_positions"].detach().cpu().tolist()
            )
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
                batch_em = qa_exact_match(
                    accum["start_preds"],
                    accum["end_preds"],
                    accum["start_labels"],
                    accum["end_labels"],
                )

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/exact_match": batch_em,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )
                loop.set_postfix(
                    {
                        "loss": avg_loss,
                        "em": batch_em,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                accum = {
                    "loss": 0.0,
                    "start_preds": [],
                    "end_preds": [],
                    "start_labels": [],
                    "end_labels": [],
                    "count": 0,
                }

        model.eval()
        eval_losses = []
        eval_start_preds = []
        eval_end_preds = []
        eval_start_labels = []
        eval_end_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                eval_loss, logits = forward_loss(model, batch)
                start_logits, end_logits = logits
                
                eval_losses.append(eval_loss.item())
                eval_start_preds.extend(start_logits.argmax(dim=-1).cpu().tolist())
                eval_end_preds.extend(end_logits.argmax(dim=-1).cpu().tolist())
                eval_start_labels.extend(batch["start_positions"].cpu().tolist())
                eval_end_labels.extend(batch["end_positions"].cpu().tolist())

        avg_eval_loss = mean(eval_losses)
        eval_em = qa_exact_match(
            eval_start_preds, 
            eval_end_preds, 
            eval_start_labels, 
            eval_end_labels
        )
        
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/exact_match": eval_em,
                "eval/epoch": epoch + 1,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.4f} - Eval Exact Match: {eval_em:.4f}"
        )

        save_checkpoint(model, tokenizer, args["output_dir"], epoch + 1)
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
