"""Collator for `RLPromptDataset` — keeps prompts/answers as plain Python lists.

vLLM consumes raw strings, and tokenisation into tensors only happens *after*
generation in `tokenization.prepare_training_batch`, so there is no padding to
do at this stage.
"""


def collate_fn(batch):
    return {
        "prompts": [example["prompt"] for example in batch],
        "answers": [example["answer"] for example in batch],
    }
