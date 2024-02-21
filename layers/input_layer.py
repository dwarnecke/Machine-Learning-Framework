"""
Input layer to the machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np
from layers.layer import Layer


class InputLayer(Layer):
    def __init__(self, input_dimension: int):
        """
        Create the input layer to the machine learning model. Each model
        must place an input layer at the start and only at the start.
        :param input_dimension: The feature dimension of the model inputs
        """

        # Call the super class initializer
        super().__init__(False)

        # Define the layer feature dimension
        self.features = input_dimension

    def forward(self, model_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. The inputs to the model are the
        simply returned after being verified.
        :param model_inputs: The inputs to the machine learning model
        :return: The same inputs of the machine learning model
        """

        # Check that the input is a two-dimensional numpy array
        if type(model_inputs) != np.ndarray:
            raise TypeError("Model inputs must be a numpy array.")
        elif np.ndim(model_inputs) != 2:
            raise ValueError("Model inputs must be two dimensional.")

        # Check that the number of inputs is consistent
        if model_inputs.shape[-1] != self.features:
            raise ValueError("Feature dimension must be consistent.")

        return model_inputs  # Return the model inputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. The gradients are practically
        useless at this point as all parameters have been updated accordingly.
        :param output_gradients: The loss gradients respecting the inputs
        :return: The same gradients of the inputs
        """

        # Check that the inputs match the number of units
        if output_gradients.shape[-1] != self.features:
            raise ValueError("Feature dimension must be consistent.")

        return output_gradients  # Return the feature loss gradients
