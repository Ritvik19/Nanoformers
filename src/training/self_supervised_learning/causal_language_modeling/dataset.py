import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        return_special_tokens_mask=False,
        add_special_tokens=False,
    )


def group_texts(batch, block_size, stride, tokenizer, bos_token_id):
    input_ids = []
    target_ids = []
    for token_ids in batch["input_ids"]:
        for index in range(0, len(token_ids) - 1, stride):
            input_chunk = token_ids[index : index + block_size]
            if len(input_chunk) != block_size:
                delta_input = block_size - len(input_chunk)
                input_chunk = input_chunk + [tokenizer.pad_token_id] * delta_input

            input_chunk = [bos_token_id] + input_chunk
            target_chunk = input_chunk.copy()
            target_chunk = [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in target_chunk
            ]
            input_ids.append(input_chunk)
            target_ids.append(target_chunk)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
    }


class CLMDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["target_ids"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = ["input_ids", "labels"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
