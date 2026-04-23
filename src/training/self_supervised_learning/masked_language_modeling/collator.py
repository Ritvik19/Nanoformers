import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, tokenizer, mlm_probability=0.15):
    input_ids = [example["input_ids"] for example in batch]
    special_tokens_mask = [example["special_tokens_mask"] for example in batch]

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    special_tokens_mask_padded = pad_sequence(
        special_tokens_mask,
        batch_first=True,
        padding_value=1,
    )
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    labels = input_ids_padded.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask_padded.bool(), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100

    input_ids_masked = input_ids_padded.clone()

    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids_masked[indices_replaced] = tokenizer.mask_token_id

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_tokens = torch.randint(
        low=0,
        high=len(tokenizer),
        size=labels.shape,
        dtype=input_ids_masked.dtype,
    )
    input_ids_masked[indices_random] = random_tokens[indices_random]

    return {
        "input_ids": input_ids_masked,
        "attention_mask": attention_mask,
        "labels": labels,
    }
