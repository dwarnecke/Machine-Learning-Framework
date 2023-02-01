"""
Tanh delineating activation function

tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


def calculate(rads: np.ndarray) -> np.ndarray:
    """
    Calculate activation values using the tanh activation function.
    :param rads: The values to be input into the function
    :type rads: np.ndarray
    :return: The respective activation values of the tanh function
    """
    return (np.exp(rads) - np.exp(-rads)) / (np.exp(rads) + np.exp(-rads))


def differentiate(rads: np.ndarray) -> np.ndarray:
    """
    Differentiate the tanh function at specified inputs.
    :param rads: The inputs at which to differentiate the function at
    :type rads: np.ndarray
    :return: The point derivatives of the activation function at the inputs
    """
    return 1 - np.square(calculate(rads))
