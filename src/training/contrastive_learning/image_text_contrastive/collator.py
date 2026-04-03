def collate_fn(batch, processor, max_length):
    images = [example["image"] for example in batch]
    texts = [example["text"] for example in batch]

    encoding = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        "pixel_values": encoding["pixel_values"],
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }
