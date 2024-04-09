"""
Dropout layer for regularizing the machine learning model.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.layers import Layer


class Dropout(Layer):
    """
    Dropout regularization layer for a neural network. These layers only
    function in model training as they randomly zero out features in
    to prevent data over-fitting.
    """

    # Define the number of dropout layers created
    layers = 0

    def __init__(self, rate: float):
        """
        Create a dropout regularization layer for a neural network.
        :param rate: The rate at which features are dropped out at
        """

        # Call the super class initializer
        super().__init__()

        # Define the layer identification key
        Dropout.layers += 1
        self.LAYER_ID = 'Dropout' + str(Dropout.layers)

        # Check and define the dropout rate
        if not 0 < rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1.")
        self._RATE = rate

        # Define the dropout filter retainer and random generator
        self._generator = np.random.default_rng()
        self._dropout_filter = None

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to be possibly zeroed out
        :return: The layer inputs after dropping units.
        """

        # Check that the training argument was passed
        try:
            in_training = kwargs['in_training']
        except KeyError:
            raise ValueError("Key word argument 'in_training' must be passed.")

        # Filter units only when in training
        if in_training:
            # Filter random units with dropout
            filter_values = self._generator.random(size=input_layers.shape)
            dropout_filter = self._RATE < filter_values
            filtered_inputs = np.where(dropout_filter, input_layers, 0)
            output_layers = filtered_inputs / (1 - self._RATE)

            # Save the filter for later use
            self._dropout_filter = dropout_filter
        else:
            output_layers = input_layers

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss gradients respecting the outputs
        :return: The partial loss derivatives with respect to layer inputs
        """

        # Check that forward propagation has been completed
        dropout_filter = self._dropout_filter[:]  # None cannot be indexed
        self._dropout_filter = None  # Erase the filter to prevent repeat use

        # Calculate the loss gradients respecting the inputs
        rescaled_grads = output_grads / (1 - self._RATE)
        input_grads = np.where(dropout_filter, rescaled_grads, 0)

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        layer = {'type': 'dropout', 'dropout_rate': self._RATE}

        return layer