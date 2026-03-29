import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        ).logits
        loss = F.cross_entropy(logits.float(), batch["labels"])

    return loss, logits
