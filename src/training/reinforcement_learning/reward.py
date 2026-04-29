"""Outcome-based reward scoring using `\\boxed{}` extraction + math_verify."""

from math_verify import parse, verify


def extract_boxed_answer(text):
    if text is None:
        return None
    parsed = parse(text)
    if not parsed:
        return None
    return parsed


def compute_outcome_rewards(completions, ground_truths):
    # Returns a list of binary floats (1.0 / 0.0), one per (completion, gt) pair.
    # `math_verify.verify` accepts either parsed expressions or raw strings;
    # parsing the ground truth as `$<value>$` gives it a fair chance of being
    # interpreted as LaTeX math when needed.
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        try:
            pred = extract_boxed_answer(completion)
            target = parse(f"${gt}$") if not isinstance(gt, list) else gt
            if not pred or not target:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if verify(target, pred) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards
