"""
Input layer to the machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np
from layers.layer import Layer


class InputLayer(Layer):
    def __init__(self, features: int):
        """
        Create the input layer to the machine learning model.
        :param features: The number of features input to the model
        """

        super().__init__(False)  # Call the super class initializer

        # Check and define the layer units
        if type(features) != int:
            raise ValueError("Units must be integers.")
        if features < 1:
            raise ValueError("Units must be greater than zero.")
        self.UNITS = features

    def forward(self, layer_inputs: np.ndarray):
        """
        Pass through this layer with forward propagation. There is no change
        in data with this layer so the inputs are practically returned as
        the outputs.
        :param layer_inputs: The inputs to the machine learning model
        :return: The same inputs of the machine learning model
        """

        # Check that the input is a two-dimensional numpy array
        if type(layer_inputs) != np.ndarray:
            raise TypeError("Layer inputs must be a numpy array.")
        if np.ndim(layer_inputs) != 2:
            raise ValueError("Layer inputs must be two dimensional.")

        # Check that the inputs match the number of units
        if layer_inputs.shape[1] != self.UNITS:
            raise ValueError("Number of inputs must be consistent.")

        return layer_inputs

    def backward(self, output_gradients: np.ndarray):
        """
        Pass through this layer with backpropagation. There are no changes
        in the data with layer so the input gradients are practically
        returned as the outputs.
        :param output_gradients: The loss gradients respecting the inputs
        :return: The same gradients of the inputs
        """

        # Check that the input is a two-dimensional numpy array
        if type(output_gradients) != np.ndarray:
            raise TypeError("Output gradients must be a numpy array.")
        if np.ndim(output_gradients) != 2:
            raise ValueError("Output gradients must be two dimensional.")

        # Check that the inputs match the number of units
        if output_gradients.shape[1] != self.UNITS:
            raise ValueError("Number of input features must be consistent.")

        return output_gradients  # Return feature loss gradients
