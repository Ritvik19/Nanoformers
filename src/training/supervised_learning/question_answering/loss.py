import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
            return_dict=True,
        )
        loss = outputs.loss
        logits = (outputs.start_logits, outputs.end_logits)

    return loss, logits
