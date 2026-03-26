import torch


def forward_loss(model, batch):
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    return outputs.loss
