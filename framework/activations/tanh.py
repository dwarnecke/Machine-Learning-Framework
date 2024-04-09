"""
Hyperbolic tangent function for delineating values.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.activations.activation import Activation

class Tanh(Activation):
    """
    Hyperbolic tangent activation function for a model layer. This range of
    this function is (-1, 1) with x < y implying f(x) < f(y).
    """

    def __init__(self, clip: int = 100):
        """
        Create a tanh activation function for a neural network layer.
        :param clip: The clipping value to prevent exponential overflow
        """

        # Check that the clipping value is positive
        if clip <= 0:
            raise ValueError("Clipping value must be positive.")

        self._CLIP = clip

    def calculate(self, radians: ndarray) -> ndarray:
        """
        Calculate the tanh activation values of many inputs.
        :param radians: The radian inputs to the tanh function
        :return: The tanh activation values of the inputs
        """

        clippings = np.clip(radians, -self._CLIP, self._CLIP)
        activations = (
            (np.exp(clippings) - np.exp(-clippings))
            / (np.exp(clippings) + np.exp(-clippings)))

        return activations

    def differentiate(self, radians: ndarray) -> ndarray:
        """
        Differentiate the tanh function at many inputs.
        :param radians: The radian inputs to the tanh function
        :return: The tanh point derivatives of the inputs
        """

        derivatives = 1 - np.square(self.calculate(radians))

        return derivatives

    def serialize(self) -> dict:
        """
        Serialize the activation function into a transmittable form.
        :return: The activation function parameters
        """

        function = {'type': 'tanh', 'clip': self._CLIP}

        return function