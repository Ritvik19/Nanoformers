import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        return_special_tokens_mask=False,
        add_special_tokens=False,
    )


def group_texts(batch, block_size, tokenizer):
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    chunk_size = block_size - 2

    input_ids = []
    special_tokens_mask = []
    for token_ids in batch["input_ids"]:
        for index in range(0, len(token_ids), chunk_size):
            input_chunk = token_ids[index : index + chunk_size]
            pad_length = chunk_size - len(input_chunk)

            wrapped_chunk = [cls_token_id] + input_chunk + [sep_token_id]
            chunk_special = [1] + [0] * len(input_chunk) + [1]

            if pad_length > 0:
                wrapped_chunk = wrapped_chunk + [pad_token_id] * pad_length
                chunk_special = chunk_special + [1] * pad_length

            input_ids.append(wrapped_chunk)
            special_tokens_mask.append(chunk_special)

    return {
        "input_ids": input_ids,
        "special_tokens_mask": special_tokens_mask,
    }


class MLMDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "special_tokens_mask": torch.tensor(
                item["special_tokens_mask"], dtype=torch.long
            ),
        }

    def __repr__(self):
        feature_list = ["input_ids", "special_tokens_mask"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
