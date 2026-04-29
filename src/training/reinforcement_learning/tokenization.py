"""Post-generation tokenization for RL training batches.

Given lists of (prompt_text, completion_text) pairs we produce padded tensors
ready for a forward pass on the policy/ref models. The `completion_mask` marks
which positions correspond to model-generated tokens (vs. the prompt prefix or
right-padding) so the loss can be restricted to just the rollout span.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def prepare_training_batch(tokenizer, prompts, completions, max_length, device):
    input_ids_list = []
    completion_mask_list = []
    labels_list = []

    for prompt_text, completion_text in zip(prompts, completions):
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completion_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        full_ids = (prompt_ids + completion_ids)[:max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))
        completion_len = len(full_ids) - prompt_len

        completion_mask = [0.0] * prompt_len + [1.0] * completion_len
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        completion_mask_list.append(torch.tensor(completion_mask, dtype=torch.float))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    completion_mask = pad_sequence(
        completion_mask_list, batch_first=True, padding_value=0.0
    )
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "completion_mask": completion_mask.to(device),
        "labels": labels.to(device),
    }
