"""
Utilities file for excess functions needed for the project.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray


def make_one_hot(labels: ndarray, classes: int) -> ndarray:
    """
    Get the respective one hot matrix out of a label vector.
    :param labels: The labels to be encoded in one hot form
    :param classes: The number of different label classes there are
    :return: The one hot representation of the labels vector
    """

    # Check that every label is a positive integer
    if np.sum(labels % 1) != 0:
        raise TypeError("Labels must be integers.")
    elif np.min(labels) < 0:
        raise ValueError("Labels must be positive.")

    # Check that the number of labels is positive
    elif classes < 1:
        raise ValueError("There must be one or more labels.")

    # Check that the maximum label is less than the number of classes
    elif np.max(labels) >= classes:
        raise ValueError("Labels must be less than the number of classes.")

    # Create the one hot matrix for the labels
    one_hot_labels = np.zeros((len(labels), classes))
    for label_idx, label_value in enumerate(labels):
        one_hot_labels[label_idx, label_value] = 1

    return one_hot_labels
