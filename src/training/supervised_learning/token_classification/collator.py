import torch


def collate_fn(batch, tokenizer):
    model_inputs = [
        {
            key: value.tolist()
            for key, value in example.items()
            if key != "labels"
        }
        for example in batch
    ]
    collated_batch = tokenizer.pad(
        model_inputs,
        padding=True,
        return_tensors="pt",
    )

    max_length = collated_batch["input_ids"].size(1)
    labels = torch.full(
        (len(batch), max_length),
        fill_value=-100,
        dtype=torch.long,
    )
    for index, example in enumerate(batch):
        example_labels = example["labels"]
        label_length = example_labels.size(0)
        if tokenizer.padding_side == "left":
            labels[index, max_length - label_length :] = example_labels
        else:
            labels[index, :label_length] = example_labels

    collated_batch["labels"] = labels
    return collated_batch
