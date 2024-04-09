"""
Input layer to the machine learning model.
"""

__author__ = 'Dylan Warnecke'

from numpy import ndarray
from framework.layers import Layer


class Input(Layer):
    """
    Input layer for a neural network. These layers are used to verify model
    input shapes and must and only come at the start of each network.
    """

    # Define the number of input layers created
    layers = 0

    def __init__(self, input_shape: tuple):
        """
        Create the input layer to the machine learning model.
        :param input_shape: The shape of the model inputs
        """

        # Call the super class initializer
        super().__init__()

        # Define the network identification key
        Input.layers += 1
        self.LAYER_ID = 'Input' + str(Input.layers)

        # Check that the input shape dimensions are positive
        for dimension in input_shape:
            if dimension <= 0:
                raise ValueError("Input dimensions must be positive.")

        # Define the model input shape
        self.INPUTS_SHAPE = input_shape

    def forward(self, model_inputs: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param model_inputs: The inputs to the machine learning model
        :return: The same inputs of the machine learning model
        """

        # Check that the model input size is consistent
        if model_inputs.shape[1:] != self.INPUTS_SHAPE:
            raise ValueError("Model input shape must be consistent.")

        # Input layer performs no transformations
        output_layers = model_inputs

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss gradients respecting the inputs
        :return: The same gradients of the inputs
        """

        # Check that the input gradients size is consistent.
        if output_grads.shape[1:] != self.INPUTS_SHAPE:
            raise ValueError("Model input shape must be consistent.")

        # Input layer performs no transformations
        input_grads = output_grads

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        layer = {'type': 'input', 'shape_in': self.INPUTS_SHAPE}

        return layer