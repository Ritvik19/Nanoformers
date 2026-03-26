import math


def mean(values):
    return sum(values) / len(values)


def perplexity_from_loss(loss):
    return math.exp(loss)
