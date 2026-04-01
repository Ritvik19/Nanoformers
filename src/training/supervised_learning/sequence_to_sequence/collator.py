from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    padded_attention_mask = pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )
    padded_labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }
