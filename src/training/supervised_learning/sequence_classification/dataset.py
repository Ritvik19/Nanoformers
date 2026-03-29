import torch
from torch.utils.data import Dataset


def build_label_mappings(dataset):
    label_feature = dataset.features.get("label")
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_to_id = {index: index for index in range(len(label_feature.names))}
        id_to_label = {index: name for index, name in enumerate(label_feature.names)}
        return label_to_id, id_to_label

    label_values = dataset.unique("label")
    try:
        label_values = sorted(label_values)
    except TypeError:
        pass

    label_to_id = {label: index for index, label in enumerate(label_values)}
    id_to_label = {index: str(label) for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def encode_label(example, label_to_id):
    return {"label": label_to_id[example["label"]]}


def tokenize_function(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=False,
    )


class SequenceClassificationDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = ["input_ids", "attention_mask", "labels"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
