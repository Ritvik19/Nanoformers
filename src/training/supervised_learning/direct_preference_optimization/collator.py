from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer):
    chosen_input_ids = [example["chosen_input_ids"] for example in batch]
    chosen_target_ids = [example["chosen_target_ids"] for example in batch]
    rejected_input_ids = [example["rejected_input_ids"] for example in batch]
    rejected_target_ids = [example["rejected_target_ids"] for example in batch]

    chosen_input_ids_padded = pad_sequence(
        chosen_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    chosen_target_ids_padded = pad_sequence(
        chosen_target_ids,
        batch_first=True,
        padding_value=-100,
    )
    rejected_input_ids_padded = pad_sequence(
        rejected_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    rejected_target_ids_padded = pad_sequence(
        rejected_target_ids,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "chosen_input_ids": chosen_input_ids_padded,
        "chosen_attention_mask": (chosen_input_ids_padded != tokenizer.pad_token_id).long(),
        "chosen_target_ids": chosen_target_ids_padded,
        "rejected_input_ids": rejected_input_ids_padded,
        "rejected_attention_mask": (rejected_input_ids_padded != tokenizer.pad_token_id).long(),
        "rejected_target_ids": rejected_target_ids_padded,
    }
