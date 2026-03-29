import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(
            **model_inputs,
            return_dict=True,
        ).logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )

    return loss, logits
