"""
Softmax layer in a multi class classification neural network.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.layers import Layer


class Softmax(Layer):
    """
    Softmax layer for a neural network. These layers create new
    probabilities features out of the feature vectors using exponential
    sums.
    """

    # Define the number of softmax layers created
    layers = 0

    def __init__(self):
        """
        Create a softmax layer for categorical classification.
        """

        # Call the superclass initializer
        super().__init__()

        # Set the network identification key
        Softmax.layers += 1
        self.LAYER_ID = 'Softmax' + str(Softmax.layers)

        # Define the layer calculation retainers
        self._input_layers = None
        self._output_layers = None

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to this softmax model layer
        :return: The softmax normalized values of the input along the axis
        """

        # Use the softmax function to convert logits to probability
        medium_odds = np.exp(input_layers)
        output_layers = medium_odds / medium_odds.sum(axis=1, keepdims=True)

        # Cache the outputs for backpropagation later
        self._input_layers = input_layers
        self._output_layers = output_layers

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss derivatives respecting outputs
        :return: The partial loss derivatives with respect to layer inputs
        """

        # Check that forward propagation has been completed
        input_layers = self._input_layers[:] # None cannot be indexed
        output_layers = self._output_layers[:]
        self._input_layers = None  # Erase the retainers to prevent repeat use
        self._output_layers = None

        # Define the rank three softmax gradients
        samples = input_layers.shape[0]
        logits = input_layers.shape[1]
        softmax_grads = np.zeros((samples, logits, logits))

        # Calculate the partial softmax derivatives respecting each logit
        for max_idx in range(logits):
            for logit_idx in range(max_idx, logits):
                if logit_idx != max_idx:
                    # Make use of the fact the matrices are symmetrical
                    off_diagonal_grads = (
                            -output_layers[:, max_idx]
                            * output_layers[:, logit_idx])
                    softmax_grads[:, max_idx, logit_idx] = off_diagonal_grads
                    softmax_grads[:, logit_idx, max_idx] = off_diagonal_grads
                else:
                    diagonal_grads = (
                            output_layers[:, logit_idx]
                            * (1 - output_layers[:, logit_idx]))
                    softmax_grads[:, max_idx, logit_idx] = diagonal_grads

        # Calculate the input gradients
        input_grads = np.squeeze(output_grads[:, np.newaxis] @ softmax_grads)

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        layer = {'type': 'softmax'}

        return layer
