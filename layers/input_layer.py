"""
Input layer to the machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np
from layers.layer import Layer


class InputLayer(Layer):
    def __init__(self, units: int):
        """
        Create the input layer to the machine learning model. Each model
        must place an input layer at the start and only at the start.
        :param units: The number of features input to the model
        """

        super().__init__(False)  # Call the super class initializer

        # Check and define the layer units
        if type(units) != int:
            raise ValueError("Units must be integers.")
        if units < 1:
            raise ValueError("Units must be greater than zero.")
        self.UNITS = units

    def forward(
            self,
            model_inputs: np.ndarray,
            in_training: bool) -> np.ndarray:
        """
        Pass through this input layer in forward propagation. The inputs to
        the model are the simply returned after being verified.
        :param model_inputs: The inputs to the machine learning model
        :param in_training: If the model is currently being trained
        :return: The same inputs of the machine learning model
        """

        # Check that the input is a two-dimensional numpy array
        if type(model_inputs) != np.ndarray:
            raise TypeError("Model inputs must be a numpy array.")
        if np.ndim(model_inputs) != 2:
            raise ValueError("Model inputs must be two dimensional.")

        # Check that the number of inputs is consistent
        if model_inputs.shape[1] != self.UNITS:
            raise ValueError("Number of inputs must be consistent.")

        return model_inputs  # Return the model inputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Pass through this input layer in backward propagation. The gradients
        are practically useless at this point as all parameters have been
        updated accordingly.
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
