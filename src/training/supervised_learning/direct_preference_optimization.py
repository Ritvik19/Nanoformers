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
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.parse_config import parse_args
from utils_direct_preference_optimization import tokenize_function, group_texts, CLMDataset, collate_fn

def load_model_and_tokenizer(args):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args["model_path"])
    model = AutoModelForCausalLM.from_pretrained(args["model_path"])
    model.to(args["device"])

    # Create frozen reference model for DPO (kept on device but in eval mode, grads off)
    ref_model = AutoModelForCausalLM.from_pretrained(args["model_path"])
    ref_model.to(args["device"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Model, reference model and tokenizer loaded...")
    return model, ref_model, tokenizer

def load_and_prepare_dataset(args, tokenizer):
    print("Loading and preparing dataset...")    
    raw_ds = HFDataset.from_json(args["dataset_path"])

    print("Tokenizing dataset...")
    lm_datasets = raw_ds.map(lambda x: tokenize_function(x, tokenizer), remove_columns=raw_ds.column_names, num_proc=256)
    
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

def train(args, model, ref_model, tokenizer, train_loader, eval_loader, optimizer, scaler, lr_scheduler):
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
            "dpo_beta": args["dpo_beta"],
        },
    )

    device = args["device"]
    output_dir = args["output_dir"]
    num_epochs = args["num_epochs"]
    gradient_accumulation_steps = args["gradient_accumulation_steps"]
    beta = args["dpo_beta"]
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}
            chosen_ids = batch["chosen_input_ids"]
            chosen_attn = batch["chosen_attention_mask"]
            chosen_labels = batch["chosen_target_ids"]
            rejected_ids = batch["rejected_input_ids"]
            rejected_attn = batch["rejected_attention_mask"]
            rejected_labels = batch["rejected_target_ids"]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_attn, return_dict=True).logits
                rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_attn, return_dict=True).logits

                with torch.no_grad():
                    ref_chosen_logits = ref_model(input_ids=chosen_ids, attention_mask=chosen_attn, return_dict=True).logits
                    ref_rejected_logits = ref_model(input_ids=rejected_ids, attention_mask=rejected_attn, return_dict=True).logits
                
                chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
                rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
                ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
                ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

                # def per_example_sum_logprob(log_probs, labels):
                #     labels_safe = labels.clone()
                #     labels_safe[labels_safe == -100] = 0  
                #     token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
                #     mask = labels.ne(-100).float()  
                #     sum_logp = (token_logp * mask).sum(dim=1)  
                #     return sum_logp

                def per_example_sum_logprob(log_probs, labels):
                    seq_len = log_probs.size(1)
                    labels = labels[:, :seq_len]

                    labels_safe = labels.clone()
                    labels_safe[labels_safe == -100] = 0
                    token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
                    token_logp = token_logp * labels.ne(-100)
                    return token_logp.sum(dim=1)

                pi_chosen = per_example_sum_logprob(chosen_log_probs, chosen_labels)
                pi_rejected = per_example_sum_logprob(rejected_log_probs, rejected_labels)
                ref_pi_chosen = per_example_sum_logprob(ref_chosen_log_probs, chosen_labels)
                ref_pi_rejected = per_example_sum_logprob(ref_rejected_log_probs, rejected_labels)

                d = (pi_chosen - pi_rejected) - (ref_pi_chosen - ref_pi_rejected)
                logits_for_sigmoid = beta * d

                loss = -F.logsigmoid(logits_for_sigmoid).mean()
                loss_for_backprop = loss / gradient_accumulation_steps

            scaler.scale(loss_for_backprop).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

                global_step += 1

                # Log training metrics to wandb
                avg_advantage = d.mean().item()
                wandb.log({
                    "train/dpo_loss": loss.item(),
                    "train/avg_advantage": avg_advantage,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                }, step=global_step)

                loop.set_postfix({
                    "dpo_loss": loss.item(),
                    "adv": avg_advantage,
                    "lr": lr_scheduler.get_last_lr()[0]
                })

        # ---- Evaluation at end of epoch ----
        model.eval()
        eval_losses = []
        eval_advantages = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # forward pass same as above but no grad and aggregated
                chosen_logits = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], return_dict=True).logits
                rejected_logits = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], return_dict=True).logits
                ref_chosen_logits = ref_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], return_dict=True).logits
                ref_rejected_logits = ref_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], return_dict=True).logits

                chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
                rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
                ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
                ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

                def per_example_sum_logprob_no_grad(log_probs, labels):
                    seq_len = log_probs.size(1)
                    labels = labels[:, :seq_len]

                    labels_safe = labels.clone()
                    labels_safe[labels_safe == -100] = 0

                    token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
                    token_logp = token_logp * labels.ne(-100)
                    return token_logp.sum(dim=1)

                pi_chosen = per_example_sum_logprob_no_grad(chosen_log_probs, batch["chosen_target_ids"])
                pi_rejected = per_example_sum_logprob_no_grad(rejected_log_probs, batch["rejected_target_ids"])
                ref_pi_chosen = per_example_sum_logprob_no_grad(ref_chosen_log_probs, batch["chosen_target_ids"])
                ref_pi_rejected = per_example_sum_logprob_no_grad(ref_rejected_log_probs, batch["rejected_target_ids"])

                d = (pi_chosen - pi_rejected) - (ref_pi_chosen - ref_pi_rejected)
                logits_for_sigmoid = beta * d
                eval_loss = -F.logsigmoid(logits_for_sigmoid).mean().item()
                eval_losses.append(eval_loss)
                eval_advantages.append((pi_chosen - pi_rejected).mean().item())

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        avg_eval_adv = sum(eval_advantages) / len(eval_advantages)

        wandb.log({
            "eval/dpo_loss": avg_eval_loss,
            "eval/avg_advantage": avg_eval_adv,
            "eval/epoch": epoch + 1,
        }, step=global_step)
        print(f"Epoch {epoch+1} — Eval DPO Loss: {avg_eval_loss:.6f} — Eval AvgAdv: {avg_eval_adv:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
    
    wandb.finish()
    print("Training finished.")


def main():
    args = parse_args()
    load_dotenv(args["env_path"])
    model, ref_model, tokenizer = load_model_and_tokenizer(args)
    train_loader, eval_loader = load_and_prepare_dataset(args, tokenizer)
    optimizer, scaler, lr_scheduler = prepare_optimizer_scaler_and_scheduler(args, model, train_loader)
    train(args, model, ref_model, tokenizer, train_loader, eval_loader, optimizer, scaler, lr_scheduler)

if __name__ == "__main__":
    main()
