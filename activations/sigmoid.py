"""
Sigmoid delineating activation function.

sigmoid(x) = 1 / (1 + exp(-x))
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


def calculate(logits: np.ndarray, clip=700) -> np.ndarray:
    """
    Calculate activation values using the sigmoid activation function.
    :param logits: The values to be input into the sigmoid function
    :param clip: The value to clip the logits at to prevent overflow
    :return: The sigmoid outputs of the respective inputs
    """
    return 1 / (1 + np.exp(np.clip(-logits, -clip, clip)))


def differentiate(logits: np.ndarray, clip=700) -> np.ndarray:
    """
    Differentiate the sigmoid function at specified input points.
    :param logits: The values to differentiate the function at
    :param clip: The value to clip the logits at to prevent overflow
    :return: The point derivatives of the logits input
    """
    return calculate(logits, clip) * (1 - calculate(logits, clip))
