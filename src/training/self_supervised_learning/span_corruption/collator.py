import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


NUM_SENTINELS = 100


def _random_segmentation(num_items, num_segments, rng):
    # Randomly split `num_items` positions into `num_segments` contiguous
    # non-empty segments by choosing `num_segments - 1` break-points among
    # the `num_items - 1` internal gaps.
    mask_indices = np.arange(num_items - 1) < (num_segments - 1)
    rng.shuffle(mask_indices)
    first_in_segment = np.pad(mask_indices, [[1, 0]])
    segment_id = np.cumsum(first_in_segment)
    _, segment_length = np.unique(segment_id, return_counts=True)
    return segment_length


def random_spans_noise_mask(length, noise_density, mean_span_length, rng):
    # Standard T5 span corruption mask (see the T5 paper, appendix B).
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(num_noise_tokens / mean_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans, rng
    )

    interleaved = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved)[:-1]
    span_start_indicator = np.zeros(length, dtype=np.int64)
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = (span_num % 2) == 1
    return is_noise


def _build_sentinel_sequence(token_ids, is_target_span, sentinel_ids, eos_token_id):
    # For each contiguous run of True positions in `is_target_span`, replace
    # the first position with the next sentinel id and drop every other
    # position. Non-target positions are kept as-is.
    length = len(token_ids)
    output = []
    span_idx = 0
    position = 0
    while position < length:
        if is_target_span[position]:
            output.append(sentinel_ids[span_idx])
            span_idx += 1
            while position < length and is_target_span[position]:
                position += 1
        else:
            output.append(int(token_ids[position]))
            position += 1
    output.append(eos_token_id)
    return output


def collate_fn(batch, tokenizer, noise_density=0.15, mean_span_length=3.0):
    sentinel_ids = [
        tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        for i in range(NUM_SENTINELS)
    ]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    rng = np.random.default_rng()

    input_tensors = []
    label_tensors = []
    for example in batch:
        token_ids = example["input_ids"].tolist()
        length = len(token_ids)

        is_noise = random_spans_noise_mask(
            length=length,
            noise_density=noise_density,
            mean_span_length=mean_span_length,
            rng=rng,
        )

        input_sequence = _build_sentinel_sequence(
            token_ids=token_ids,
            is_target_span=is_noise,
            sentinel_ids=sentinel_ids,
            eos_token_id=eos_token_id,
        )
        target_sequence = _build_sentinel_sequence(
            token_ids=token_ids,
            is_target_span=~is_noise,
            sentinel_ids=sentinel_ids,
            eos_token_id=eos_token_id,
        )

        input_tensors.append(torch.tensor(input_sequence, dtype=torch.long))
        label_tensors.append(torch.tensor(target_sequence, dtype=torch.long))

    input_ids = pad_sequence(
        input_tensors,
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        label_tensors,
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
