import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        return_special_tokens_mask=False,
        add_special_tokens=False,
    )


def group_texts(batch, block_size):
    concatenated = []
    for token_ids in batch["input_ids"]:
        concatenated.extend(token_ids)

    total_length = (len(concatenated) // block_size) * block_size

    input_ids = [
        concatenated[index : index + block_size]
        for index in range(0, total_length, block_size)
    ]
    return {"input_ids": input_ids}


class SpanCorruptionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = ["input_ids"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
