import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    start_positions = [example["start_positions"] for example in batch]
    end_positions = [example["end_positions"] for example in batch]

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
        "start_positions": torch.stack(start_positions),
        "end_positions": torch.stack(end_positions),
    }
