"""
Sigmoid activation function for delineating values.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.activations.activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function for a model layer. The range of this
    function is (0, 1) with x < y implying f(x) < f(y).
    """

    def __init__(self, clip: int = 100):
        """
        Create a sigmoid activation function for a neural network layer.
        :param clip: The clipping value to prevent exponential overflow
        """

        # Check that the clipping value is positive
        if clip <= 0:
            raise ValueError("Clipping value must be positive.")

        self._CLIP = clip

    def calculate(self, logits: ndarray) -> ndarray:
        """
        Calculate the sigmoid activation values of many inputs.
        :param logits: The logit inputs to the sigmoid function
        :return: The sigmoid activation values of the inputs
        """

        clippings = np.clip(logits, -self._CLIP, self._CLIP)
        activations = 1 / (1 + np.exp(-clippings))

        return activations

    def differentiate(self, logits: ndarray, clip: int = 100) -> ndarray:
        """
        Differentiate the sigmoid function at many inputs.
        :param logits: The logit inputs to the sigmoid function
        :param clip: The clipping value to prevent exponential overflow
        :return: The sigmoid point derivatives of the inputs
        """

        derivatives = self.calculate(logits) * (1 - self.calculate(logits))

        return derivatives

    def serialize(self) -> dict:
        """
        Serialize the activation function into a transmittable form.
        :return: The activation function parameters
        """

        function = {'type': 'sigmoid', 'clip': self._CLIP}

        return function
