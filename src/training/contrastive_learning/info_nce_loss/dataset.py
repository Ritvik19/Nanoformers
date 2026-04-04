import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer, max_length):
    anchor_tokens = tokenizer(
        example["text_a"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=False,
    )
    positive_tokens = tokenizer(
        example["text_b"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=False,
    )
    return {
        "anchor_input_ids": anchor_tokens["input_ids"],
        "anchor_attention_mask": anchor_tokens["attention_mask"],
        "positive_input_ids": positive_tokens["input_ids"],
        "positive_attention_mask": positive_tokens["attention_mask"],
    }


class InfoNCELossDataset(Dataset):
    """Positive-pair dataset for InfoNCE training.

    Expects HF dataset columns: text_a, text_b (positive pairs).
    In-batch negatives are formed automatically during loss computation.
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
            "positive_input_ids": torch.tensor(item["positive_input_ids"], dtype=torch.long),
            "positive_attention_mask": torch.tensor(item["positive_attention_mask"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = [
            "anchor_input_ids",
            "anchor_attention_mask",
            "positive_input_ids",
            "positive_attention_mask",
        ]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
