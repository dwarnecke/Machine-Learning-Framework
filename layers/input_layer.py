"""
Input layer to the machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np
from layers.layer import Layer


class InputLayer(Layer):
    """
    Input layer for a neural network. These layers are used to verify model
    input shapes and must and only come at the start of each network.
    """

    # Define the number of input layers created
    layers = 0

    def __init__(self, input_shape: tuple):
        """
        Create the input layer to the machine learning model. Each model
        must place an input layer at the start and only at the start.
        :param input_shape: The shape of the model inputs
        """

        # Call the super class initializer
        super().__init__(False)

        # Check that the input shape dimensions are positive
        for dimension in input_shape:
            if dimension <= 0:
                raise ValueError("Input dimensions must be positive.")

        # Define the network identification key
        InputLayer.layers += 1
        self._network_id = 'InputLayer' + str(InputLayer.layers)

        # Define the model input shape
        self.OUTPUT_SHAPE = input_shape

    def forward(self, model_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. The inputs to the model are the
        simply returned after being verified.
        :param model_inputs: The inputs to the machine learning model
        :return: The same inputs of the machine learning model
        """

        # Check that the model input size is consistent
        if model_inputs.shape[1:] != self.OUTPUT_SHAPE:
            raise ValueError("Model input shape must be consistent.")

        # Input layer performs no transformations
        layer_outputs = model_inputs

        return layer_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. The gradients are practically
        useless at this point as all parameters have been updated accordingly.
        :param output_gradients: The loss gradients respecting the inputs
        :return: The same gradients of the inputs
        """

        # Check that the input gradients size is consistent.
        if output_gradients.shape[1:] != self.OUTPUT_SHAPE:
            raise ValueError("Model input shape must be consistent.")

        # Input layer performs no transformations
        input_gradients = output_gradients

        return input_gradients