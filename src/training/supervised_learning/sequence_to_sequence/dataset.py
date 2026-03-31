import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer, max_length):
    model_inputs = tokenizer(
        example["source"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = tokenizer(
        example["target"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class SequenceToSequenceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = ["input_ids", "attention_mask", "labels"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
