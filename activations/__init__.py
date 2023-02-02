"""
Initialization file for the activations package.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np
from activations import relu
from activations import sigmoid
from activations import tanh


# Ensure that these values always match the module names in this package
valid_activations = ['relu', 'sigmoid', 'tanh']


def verify_activation(activation: str) -> bool:
    """
    Verify the activation function string is valid.
    :param activation: The respective activation function string to check
    :return: Whether the activation string is valid or not
    """

    # Check if the activation string is valid
    if activation in valid_activations or activation is None:
        return True
    return False


def calculate(activation: str, linear_inputs: np.ndarray) -> np.ndarray:
    """
    Use one of the many activation functions.
    :param activation: The respective string of the wanted activation function
    :param linear_inputs: The linear inputs to the activation function
    :return: The output values of the activation function
    """

    # Calculate the activation function outputs
    activated_outputs = linear_inputs
    if activation == 'relu':
        activated_outputs = relu.calculate(linear_inputs)
    elif activation == 'sigmoid':
        activated_outputs = sigmoid.calculate(linear_inputs)
    elif activation == 'tanh':
        activated_outputs = tanh.calculate(linear_inputs)
    elif activation is not None:
        raise ValueError("Activation function string is invalid.")

    return activated_outputs


def differentiate(activation: str, linear_inputs: np.ndarray) -> np.ndarray:
    """
    Differentiate one of the many activation functions.
    :param activation: The respective string of the wanted activation function
    :param linear_inputs: The values at which to differentiate the function at
    :return: The point derivatives of the respective activation function
    """

    # Calculate the activation function gradient
    activation_gradient = 1
    if activation == 'relu':
        activation_gradient = relu.differentiate(linear_inputs)
    elif activation == 'sigmoid':
        activation_gradient = sigmoid.differentiate(linear_inputs)
    elif activation == 'tanh':
        activation_gradient = tanh.differentiate(linear_inputs)
    elif activation is not None:
        raise ValueError("Activation function string is invalid.")

    return activation_gradient
