import torch
from torch.utils.data import Dataset


def tokenize_function(example, tokenizer):
    chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False)

    return {
        "chosen_ids": tokenizer(
            chosen,
            return_special_tokens_mask=False,
            add_special_tokens=False,
        )["input_ids"],
        "rejected_ids": tokenizer(
            rejected,
            return_special_tokens_mask=False,
            add_special_tokens=False,
        )["input_ids"],
        "prompt_length": len(
            tokenizer(
                prompt,
                return_special_tokens_mask=False,
                add_special_tokens=False,
            )["input_ids"]
        ),
    }


def group_texts(batch, block_size, tokenizer):
    chosen_input_ids = []
    rejected_input_ids = []
    chosen_target_ids = []
    rejected_target_ids = []

    for chosen, rejected, prompt_length in zip(
        batch["chosen_ids"], batch["rejected_ids"], batch["prompt_length"]
    ):
        chosen_chunk = chosen[:block_size]
        if len(chosen_chunk) != block_size:
            delta_input = block_size - len(chosen_chunk)
            chosen_chunk = chosen_chunk + [tokenizer.pad_token_id] * delta_input

        chosen_target = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in chosen_chunk
        ]
        chosen_target[:prompt_length] = [-100] * prompt_length

        rejected_chunk = rejected[:block_size]
        if len(rejected_chunk) != block_size:
            delta_input = block_size - len(rejected_chunk)
            rejected_chunk = rejected_chunk + [tokenizer.pad_token_id] * delta_input

        rejected_target = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in rejected_chunk
        ]
        rejected_target[:prompt_length] = [-100] * prompt_length

        chosen_input_ids.append(chosen_chunk)
        chosen_target_ids.append(chosen_target)
        rejected_input_ids.append(rejected_chunk)
        rejected_target_ids.append(rejected_target)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_target_ids": chosen_target_ids,
        "rejected_input_ids": rejected_input_ids,
        "rejected_target_ids": rejected_target_ids,
    }


class DPODataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "chosen_input_ids": torch.tensor(item["chosen_input_ids"], dtype=torch.long),
            "chosen_target_ids": torch.tensor(item["chosen_target_ids"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(item["rejected_input_ids"], dtype=torch.long),
            "rejected_target_ids": torch.tensor(item["rejected_target_ids"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = [
            "chosen_input_ids",
            "chosen_target_ids",
            "rejected_input_ids",
            "rejected_target_ids",
        ]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
