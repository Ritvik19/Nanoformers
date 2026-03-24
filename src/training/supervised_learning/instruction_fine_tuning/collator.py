from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    input_ids = [example["input_ids"] for example in batch]
    labels = [example["labels"] for example in batch]

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }
