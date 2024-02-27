"""
Dropout layer for regularizing the machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np
from layers.layer import Layer


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
        super().__init__(False)

        # Check and define the dropout rate
        if not 0 < rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1.")
        self._RATE = rate

        # Define the network identification key
        Dropout.layers += 1
        self._network_id = 'Dropout' + str(Dropout.layers)

        # Define the dropout filter retainer and random generator
        self._generator = np.random.default_rng()
        self._dropout_filter = None

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. If the model is in training,
        certain units will be dropped out and the others will be adequately
        scaled. If not, the inputs are simply returned as the outputs.
        :param layer_inputs: The inputs to be possibly zeroed out
        :return: The layer inputs after dropping units.
        """

        # Check that the training argument was passed
        try:
            in_training = kwargs['in_training']
        except KeyError:
            raise ValueError("Key word argument 'in_training' must be passed.")

        if in_training:
            # Filter certain input units with dropout
            filter_values = self._generator.random(size=layer_inputs.shape)
            dropout_filter = filter_values > self._RATE
            filtered_inputs = np.where(dropout_filter, layer_inputs, 0)
            layer_outputs = filtered_inputs / (1 - self._RATE)

            # Cache the filter for later use
            self._dropout_filter = dropout_filter
        else:
            # No dropout is used outside of training
            layer_outputs = layer_inputs

        return layer_outputs

    def backward(self, output_gradients):
        """
        Backward propagate through this layer. Gradients only will
        propagate if their respective input counterparts managed to pass the
        dropout filter.
        :param output_gradients: The loss gradients respecting the outputs
        :return: The loss gradients respecting the layer inputs
        """

        # Check that forward propagation has been completed
        if self._dropout_filter is None:
            raise ValueError("Forward propagation must be completed.")

        # Calculate the loss gradients respecting the inputs
        dropout_filter = self._dropout_filter
        rescaled_gradients = output_gradients / (1 - self._RATE)
        input_gradients = np.where(dropout_filter, rescaled_gradients, 0)

        # Clear the dropout filter to prevent double use
        self._dropout_filter = None

        return input_gradients
