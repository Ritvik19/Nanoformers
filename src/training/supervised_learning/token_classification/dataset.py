import torch
from torch.utils.data import Dataset


def _extract_label_names(label_feature):
    if hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
        return label_feature.feature.names
    if hasattr(label_feature, "names"):
        return label_feature.names
    return None


def build_label_mappings(dataset, labels_column):
    label_feature = dataset.features.get(labels_column)
    label_names = _extract_label_names(label_feature)

    if label_names is not None:
        label_to_id = {index: index for index in range(len(label_names))}
        id_to_label = {index: name for index, name in enumerate(label_names)}
        return label_to_id, id_to_label

    label_values = {
        label
        for label_sequence in dataset[labels_column]
        for label in label_sequence
    }
    try:
        label_values = sorted(label_values)
    except TypeError:
        label_values = list(label_values)

    label_to_id = {label: index for index, label in enumerate(label_values)}
    id_to_label = {index: str(label) for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def encode_labels(example, labels_column, label_to_id):
    return {"labels": [label_to_id[label] for label in example[labels_column]]}


def tokenize_and_align_labels(
    example,
    tokenizer,
    tokens_column,
    max_length,
    label_all_tokens=False,
):
    if len(example[tokens_column]) != len(example["labels"]):
        raise ValueError(
            "Token classification examples must have the same number of tokens and labels."
        )

    tokenized_example = tokenizer(
        example[tokens_column],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        return_special_tokens_mask=False,
    )

    aligned_labels = []
    previous_word_idx = None
    try:
        word_ids = tokenized_example.word_ids()
    except ValueError as exc:
        raise ValueError(
            "Token classification requires a fast tokenizer with `word_ids()` support."
        ) from exc

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(example["labels"][word_idx])
        elif label_all_tokens:
            aligned_labels.append(example["labels"][word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    tokenized_example["labels"] = aligned_labels
    return tokenized_example


class TokenClassificationDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in item.items()
        }

    def __repr__(self):
        feature_list = list(self.ds.features.keys())
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
