"""
Linear activation function for maintaining linear transformations.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.activations.activation import Activation

class Linear(Activation):
    """
    Linear activation function for a model layer. This function returns
    the inputs as passed.
    """

    def __init__(self):
        """
        Create a linear activation function for a neural network.
        """
        pass

    def calculate(self, inputs: ndarray) -> ndarray:
        """
        Calculate the linear activation values of many inputs.
        :param inputs: The inputs to the linear function
        :return: The linear activation values of the inputs
        """

        activations = inputs

        return activations

    def differentiate(self, inputs: ndarray) -> ndarray:
        """
        Differentiate the linear function at many inputs.
        :param inputs: The inputs to the linear function
        :return: The point derivatives of the inputs
        """

        derivatives = np.ones_like(inputs)

        return derivatives

    def serialize(self) -> dict:
        """
        Serialize the activation function into a transmittable form.
        :return: The activation function parameters
        """

        function = {'type': 'linear'}

        return function


