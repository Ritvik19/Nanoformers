import torch
import torch.nn.functional as F


def forward_loss(model, batch):
    
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs.logits
        labels = batch["labels"]

        # Flatten logits and labels to compute cross-entropy loss
        # logits shape: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # labels shape: (batch, seq_len) -> (batch * seq_len)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    return loss, logits
