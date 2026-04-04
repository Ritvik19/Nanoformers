import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer, max_length):
    anchor_tokens = tokenizer(
        example["text_a"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=False,
    )
    other_tokens = tokenizer(
        example["text_b"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=False,
    )
    return {
        "anchor_input_ids": anchor_tokens["input_ids"],
        "anchor_attention_mask": anchor_tokens["attention_mask"],
        "other_input_ids": other_tokens["input_ids"],
        "other_attention_mask": other_tokens["attention_mask"],
    }


class ContrastiveLossDataset(Dataset):
    """Pair dataset with binary similarity labels.

    Expects HF dataset columns: text_a, text_b, label (1 = similar, 0 = dissimilar).
    """

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "anchor_input_ids": torch.tensor(item["anchor_input_ids"], dtype=torch.long),
            "anchor_attention_mask": torch.tensor(item["anchor_attention_mask"], dtype=torch.long),
            "other_input_ids": torch.tensor(item["other_input_ids"], dtype=torch.long),
            "other_attention_mask": torch.tensor(item["other_attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = [
            "anchor_input_ids",
            "anchor_attention_mask",
            "other_input_ids",
            "other_attention_mask",
            "labels",
        ]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
