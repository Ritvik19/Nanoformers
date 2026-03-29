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
