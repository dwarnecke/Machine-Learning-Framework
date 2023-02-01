"""
Relu delineating activation function

relu(x) = max(x, 0)
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


def calculate(inputs: np.ndarray) -> np.ndarray:
    """
    Calculate activation values using the relu activation function.
    :param inputs: The values to be input into the function
    :return: The respective activation values of the relu function
    """
    return np.maximum(inputs, 0)


def differentiate(inputs: np.ndarray) -> np.ndarray:
    """
    Differentiate the relu function at specified inputs.
    :param inputs: The inputs at which to differentiate the function at
    :return: The point derivatives of the activation function at the inputs
    """
    return np.where(inputs > 0, 1, 0)
