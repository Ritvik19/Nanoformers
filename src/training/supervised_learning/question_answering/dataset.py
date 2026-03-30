import torch
from torch.utils.data import Dataset


def tokenize_and_align_answers(example, tokenizer, max_length):
    question = example["question"].lstrip()
    
    tokenized_example = tokenizer(
        question,
        example["context"],
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offset_mapping = tokenized_example.pop("offset_mapping")
    answers = example["answers"]

    if len(answers["answer_start"]) == 0:
        tokenized_example["start_positions"] = 0
        tokenized_example["end_positions"] = 0
        return tokenized_example

    start_char = answers["answer_start"][0]
    end_char = start_char + len(answers["text"][0])
    sequence_ids = tokenized_example.sequence_ids()

    idx = 0
    try:
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
    except IndexError:
        tokenized_example["start_positions"] = 0
        tokenized_example["end_positions"] = 0
        return tokenized_example

    if offset_mapping[context_start][0] > start_char or offset_mapping[context_end][1] < end_char:
        tokenized_example["start_positions"] = 0
        tokenized_example["end_positions"] = 0
    else:
        idx = context_start
        while idx <= context_end and offset_mapping[idx][0] <= start_char:
            idx += 1
        tokenized_example["start_positions"] = idx - 1

        idx = context_end
        while idx >= context_start and offset_mapping[idx][1] >= end_char:
            idx -= 1
        tokenized_example["end_positions"] = idx + 1

    return tokenized_example


class QuestionAnsweringDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(item["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(item["end_positions"], dtype=torch.long),
        }

    def __repr__(self):
        feature_list = ["input_ids", "attention_mask", "start_positions", "end_positions"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
