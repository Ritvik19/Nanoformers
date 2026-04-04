def collate_fn(batch, tokenizer, image_processor, max_length):
    images = [example["image"] for example in batch]
    texts = [example["text"] for example in batch]

    text_encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    image_encoding = image_processor(images, return_tensors="pt")

    return {
        "pixel_values": image_encoding["pixel_values"],
        "input_ids": text_encoding["input_ids"],
        "attention_mask": text_encoding["attention_mask"],
    }
