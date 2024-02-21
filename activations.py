"""
Activation functions to delineate any linearly transformed data.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


def sigmoid(logits: np.ndarray, differentiate: bool = False) -> np.ndarray:
    """
    Calculate activation values or point derivatives using the sigmoid
    function.
    :param logits: The input logits to the sigmoid function
    :param differentiate: If the desired outputs are the point derivatives
    :return: The activation values or point derivatives of the logits
    """

    if differentiate:
        derivatives = sigmoid(logits) * (1 - sigmoid(logits))
        return derivatives

    clip = 100  # Large value to prevent exponential overflow
    activations = 1 / (1 + np.exp(np.clip(-logits, -clip, clip)))
    return activations


def tanh(radians: np.ndarray, differentiate: bool = False) -> np.ndarray:
    """
    Calculate activation values or point derivatives using the tanh function.
    :param radians: The input radians to the tanh function
    :param differentiate: If the desired outputs are the point derivatives
    :return: The activation values or point derivatives of the radians
    """

    if differentiate:
        derivatives = 1 - np.square(tanh(radians))
        return derivatives

    clip = 100  # Large value to prevent exponential overflow
    clipped_radians = np.clip(radians, -clip, clip)
    activations = (
            (np.exp(clipped_radians) - np.exp(-clipped_radians))
            / (np.exp(clipped_radians) + np.exp(-clipped_radians)))
    return activations


def relu(inputs: np.ndarray, differentiate: bool = False) -> np.ndarray:
    """
    Calculate activation values or point derivatives using the relu function.
    :param inputs: The inputs to the relu function
    :param differentiate: If the desired outputs are the point derivatives
    :return: The activation values or point derivatives of the inputs
    """

    if differentiate:
        derivatives = np.where(inputs > 0, 1, 0)
        return derivatives

    activations = np.maximum(inputs, 0)
    return activations


def linear(inputs: np.ndarray, differentiate: bool = False) -> np.ndarray:
    """
    Dummy activation function that is equivalent to no activation function.
    :param inputs: The inputs to the linear function
    :param differentiate: If the desired outputs are the point derivatives
    :return: The input values or shape-shaped array of ones
    """

    if differentiate:
        derivatives = np.ones_like(inputs)
        return derivatives

    return inputs


# Ensure that these values always match the method names in this module
valid_activations = (None, linear, relu, sigmoid, tanh)