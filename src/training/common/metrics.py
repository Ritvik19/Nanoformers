import math


def mean(values):
    return sum(values) / len(values)


def perplexity_from_loss(loss):
    return math.exp(loss)


def accuracy(predictions, labels):
    if not labels:
        return 0.0
    correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels))
    return correct / len(labels)


def masked_accuracy(predictions, labels, ignore_index=-100):
    correct = 0
    total = 0

    for prediction_sequence, label_sequence in zip(predictions, labels):
        for prediction, label in zip(prediction_sequence, label_sequence):
            if label == ignore_index:
                continue
            total += 1
            correct += int(prediction == label)

    if total == 0:
        return 0.0
    return correct / total


def qa_exact_match(start_predictions, end_predictions, start_labels, end_labels, ignore_index=-100):
    correct = 0
    total = 0

    for p_start, p_end, l_start, l_end in zip(start_predictions, end_predictions, start_labels, end_labels):
        if l_start == ignore_index or l_end == ignore_index:
            continue
        total += 1
        if p_start == l_start and p_end == l_end:
            correct += 1

    if total == 0:
        return 0.0
    return correct / total
