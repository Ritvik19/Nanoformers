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
