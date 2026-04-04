from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    anchor_input_ids = [example["anchor_input_ids"] for example in batch]
    anchor_attention_mask = [example["anchor_attention_mask"] for example in batch]
    positive_input_ids = [example["positive_input_ids"] for example in batch]
    positive_attention_mask = [example["positive_attention_mask"] for example in batch]

    return {
        "anchor_input_ids": pad_sequence(
            anchor_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "anchor_attention_mask": pad_sequence(
            anchor_attention_mask, batch_first=True, padding_value=0
        ),
        "positive_input_ids": pad_sequence(
            positive_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "positive_attention_mask": pad_sequence(
            positive_attention_mask, batch_first=True, padding_value=0
        ),
    }
