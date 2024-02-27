"""
Flattening layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import math
import numpy as np

from layers.layer import Layer


class Flatten(Layer):
    """
    Flatten layer for a neural network. This layer flattens the data for
    each example from multi-rank into single rank.
    """

    # Define the number of flatten layers created
    layers = 0

    def __init__(self):
        """
        Create a data flattening layer for a neural network.
        """

        # Call the superclass initializer
        super().__init__(False)

        # Define the network identification key
        Flatten.layers += 1
        self._network_id = 'Dense' + str(Flatten.layers)

        # Define the layer dimension shapes
        self.INPUT_SHAPE = None
        self._DIMENSION = None

    def initialize(self, input_shape: tuple) -> None:
        """
        Initialize the layer to define the new output shape.
        :param input_shape: The array shape of a single input sample
        """

        # Define the layer input and output shapes
        self.INPUT_SHAPE = input_shape
        self._DIMENSION = math.prod(input_shape)
        self.OUTPUT_SHAPE = (self._DIMENSION,)

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Inputs are flattened into a
        rank-one array for each input sample.
        :param layer_inputs: The inputs to this convolution layer
        :return: The outputs given the inputs and current parameters
        """

        # Flatten the layer inputs
        new_shape = (-1, self._DIMENSION)
        layer_outputs = np.reshape(layer_inputs, new_shape)

        return layer_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. Gradients will be reshaped
        into the original layer input shape.
        :param output_gradients: The loss gradients respecting layer outputs
        :return: The partial derivatives of the loss with respect of inputs
        """

        # Reshape the layer gradients
        new_shape = (-1) + self.INPUT_SHAPE
        input_gradients = np.reshape(output_gradients, new_shape)

        return input_gradients