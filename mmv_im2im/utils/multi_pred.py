import numpy as np


def mean_prediction(samples):
    stacked_predictions = np.array(samples)
    mean_volume = np.mean(stacked_predictions, axis=0)

    return mean_volume


def max_prediction(samples):
    stacked_predictions = np.array(samples)
    max_volume = np.max(stacked_predictions, axis=0)

    return max_volume


def variance_prediction(samples):
    stacked_predictions = np.array(samples)
    variance_volume = np.var(stacked_predictions, axis=0)
    return variance_volume


def add_prediction(samples):
    stacked_predictions = np.array(samples)
    sum_volume = np.sum(stacked_predictions, axis=0)
    return sum_volume
