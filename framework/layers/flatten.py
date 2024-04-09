"""
Flattening layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

import math
import numpy as np
from numpy import ndarray
from framework.layers import Layer


class Flatten(Layer):
    """
    Flatten layer for a neural network. These layers flatten multi-rank
    example data into a single rank.
    """

    # Define the number of flatten layers created
    layers = 0

    def __init__(self):
        """
        Create a data flattening layer for a neural network.
        """

        # Call the superclass initializer
        super().__init__()

        # Define the layer identification key
        Flatten.layers += 1
        self.LAYER_ID = 'Flatten' + str(Flatten.layers)

        # Define the layer configuration dimensions and flag
        self._INPUTS_SHAPE = None
        self._OUTPUTS_SHAPE = None
        self._is_configured = False

    def _configure(self, inputs_shape: tuple) -> bool:
        """
        Configure the layer to be used.
        :param inputs_shape: The shape of the layer input data
        :return: If the layer was successfully configured
        """

        # Define the layer input and flattened output shapes
        self._INPUTS_SHAPE = inputs_shape
        dimension = math.prod(inputs_shape)
        self._OUTPUTS_SHAPE = (dimension,)

        return True

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to this convolution layer
        :return: The outputs given the inputs and current parameters
        """

        # Check that the layer is configured
        if not self._is_configured:
            inputs_shape = input_layers.shape[1:]
            self._is_configured = self._configure(inputs_shape)

        # Flatten the layer inputs
        new_shape = (-1,) + self._OUTPUTS_SHAPE
        output_layers = np.reshape(input_layers, new_shape)

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss gradients respecting layer outputs
        :return: The partial loss derivatives with respect to layer inputs
        """

        # Reshape the layer gradients
        new_shape = (-1,) + self._INPUTS_SHAPE
        input_grads = np.reshape(output_grads, new_shape)

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        # Check that the layer is configured
        if not self._is_configured:
            raise ValueError("Layer must be configured before serializing.")

        # Collect all the layer attributes
        layer = {
            'type': 'flatten',
            'shape_in': self._INPUTS_SHAPE,
            'shape_out': self._OUTPUTS_SHAPE
        }

        return layer