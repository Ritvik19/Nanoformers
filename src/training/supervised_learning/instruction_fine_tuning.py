import math
import os
import sys
import time

import bitsandbytes as bnb
import torch
import wandb
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from itertools import cycle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.parse_config import parse_args
from utils_instruction_fine_tuning import tokenize_function, group_texts, CLMDataset, collate_fn

def load_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args["model_path"])
    model = AutoModelForCausalLM.from_pretrained(args["model_path"])
    model.to(args["device"])
    print("Model and tokenizer loaded...")
    return model, tokenizer 

def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")    
    raw_ds = HFDataset.from_json(args["dataset_path"])

    print("Tokenizing dataset...")
    lm_datasets = raw_ds.map(
        lambda x: tokenize_function(x, tokenizer), remove_columns=raw_ds.column_names, num_proc=256
    )
    
    print("Tokenized dataset:")
    print(lm_datasets)
    
    print("Splitting dataset...")
    test_size = len(lm_datasets)%1000
    split = lm_datasets.train_test_split(test_size=test_size)
    
    print("Grouping texts into blocks...")
    max_length = args["max_length"] 
    train_ds = CLMDataset(split["train"].map(
        lambda x: group_texts(x, block_size=max_length, tokenizer=tokenizer),  batched=True, num_proc=256
    ), tokenizer=tokenizer)
    eval_ds = CLMDataset(split["test"].map(
        lambda x: group_texts(x, block_size=max_length, tokenizer=tokenizer),  batched=True, num_proc=256
    ), tokenizer=tokenizer)
    print(train_ds)
    print(eval_ds)

    print("Preparing dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    eval_loader  = DataLoader(eval_ds, batch_size=args["batch_size"], shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))
    print("Dataset loaded and prepared...")
    return train_loader, eval_loader

def prepare_optimizer_scaler_and_scheduler(args, model, train_loader):
    print("Preparing optimizer, scaler, and scheduler...")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=float(args["learning_rate"]))

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args["gradient_accumulation_steps"])
    max_train_steps = args["num_epochs"] * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * max_train_steps),
        num_training_steps=max_train_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    print("Optimizer, scaler, and scheduler prepared...")
    return optimizer, scaler, lr_scheduler

def train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, lr_scheduler):
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
    output_dir = args["output_dir"]
    num_epochs = args["num_epochs"]
    gradient_accumulation_steps = args["gradient_accumulation_steps"]
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

                global_step += 1

                # Log training metrics to wandb
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                }, step=global_step)

                loop.set_postfix({
                    "loss": loss.item() * gradient_accumulation_steps,
                    "lr": lr_scheduler.get_last_lr()[0]
                })

        # ---- Evaluation at end of epoch ----
        model.eval()
        eval_losses = []
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
            eval_losses.append(outputs.loss.item())
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        perplexity = math.exp(avg_eval_loss)

        wandb.log({
            "eval/loss": avg_eval_loss,
            "eval/perplexity": perplexity,
            "eval/epoch": epoch + 1,
        }, step=global_step)
        print(f"Epoch {epoch+1} — Eval Loss: {avg_eval_loss:.4f} — Perplexity: {perplexity:.2f}")

        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
    
    wandb.finish()
    print("Training finished.")


def main():
    args = parse_args()
    load_dotenv(args["env_path"])
    model, tokenizer = load_model_and_tokenizer(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer)
    optimizer, scaler, lr_scheduler = prepare_optimizer_scaler_and_scheduler(args, model, train_loader)
    train(args, model, tokenizer, train_loader, eval_loader, optimizer, scaler, lr_scheduler)

if __name__ == "__main__":
    main()
