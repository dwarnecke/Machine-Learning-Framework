"""
Rectified linear unit activation function for delineating values.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.activations.activation import Activation

class Relu(Activation):
    """
    Rectified linear unit activation function for a model layer. This
    function is equivalent to max(x, slope * x) for all inputs x element wise.
    """

    def __init__(self, slope: float = 0):
        """
        Create a rectified linear unit activation function for a neural
        network layer.
        :param slope: The slope of the negative values in the function.
        """

        # Check that the slope is positive and less than one
        if 1 <= slope or slope < 0:
            raise ValueError("Leak slope must be less than one and positive.")

        self._SLOPE = slope

    def calculate(self, inputs: ndarray) -> ndarray:
        """
        Calculate the relu activation values of many inputs.
        :param inputs: The inputs to the leaky relu function
        :return: The relu activation values of the inputs
        """

        activations = np.fmax(inputs, self._SLOPE * inputs)

        return activations

    def differentiate(self, inputs: ndarray) -> ndarray:
        """
        Differentiate the relu function at many inputs.
        :param inputs: The inputs to the relu function
        :return: The relu point derivatives of the inputs
        """

        derivatives = np.where(inputs > 0, 1, self._SLOPE)

        return derivatives

    def serialize(self) -> dict:
        """
        Serialize the activation function into a transmittable form.
        :return: The activation function parameters
        """

        function = {'type': 'relu', 'slope': self._SLOPE}

        return function