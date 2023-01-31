"""
Utilities file for functions that don't necessarily fit anywhere else in the
project.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


def make_one_hot(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Get the respective one hot matrix out of a label vector.
    :param labels: The labels to be encoded in one hot form
    :param n_labels: The number of different labels to possibly be
    :return: The one hot representation of the labels vector
    """

    # Check that every label is a positive integer
    if np.sum(labels % 1) != 0:
        raise TypeError("Labels must be integers.")
    elif labels.min() < 0:
        raise ValueError("Labels must be positive.")

    # Check that the number of labels is positive
    elif n_labels < 1:
        raise ValueError("There must be one or more labels.")

    # Check that the maximum label is less than the number of labels
    elif labels.max() >= n_labels:
        raise ValueError("Labels must be less than the specified number.")

    # Create the one hot matrix for the labels
    one_hot_labels = np.zeros((len(labels), n_labels))
    for label_idx, label_value in enumerate(labels):
        one_hot_labels[label_idx, label_value] = 1

    return one_hot_labels  # Return the one hot version of the labels
