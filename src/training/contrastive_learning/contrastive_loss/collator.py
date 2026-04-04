import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    anchor_input_ids = [example["anchor_input_ids"] for example in batch]
    anchor_attention_mask = [example["anchor_attention_mask"] for example in batch]
    other_input_ids = [example["other_input_ids"] for example in batch]
    other_attention_mask = [example["other_attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    return {
        "anchor_input_ids": pad_sequence(
            anchor_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "anchor_attention_mask": pad_sequence(
            anchor_attention_mask, batch_first=True, padding_value=0
        ),
        "other_input_ids": pad_sequence(
            other_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "other_attention_mask": pad_sequence(
            other_attention_mask, batch_first=True, padding_value=0
        ),
        "labels": torch.stack(labels),
    }
