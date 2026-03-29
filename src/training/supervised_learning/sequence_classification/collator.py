import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask_padded = pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": torch.stack(labels),
    }
