import math

import bitsandbytes as bnb
import torch
from transformers import get_scheduler


def build_optimizer(model, learning_rate):
    return bnb.optim.AdamW8bit(model.parameters(), lr=float(learning_rate))


def build_scheduler(args, optimizer, train_loader_length):
    num_update_steps_per_epoch = math.ceil(
        train_loader_length / args["gradient_accumulation_steps"]
    )
    max_train_steps = args["num_epochs"] * num_update_steps_per_epoch
    return get_scheduler(
        name=args["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * max_train_steps),
        num_training_steps=max_train_steps,
    )


def build_grad_scaler():
    return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
