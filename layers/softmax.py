"""
Softmax layer in a multi class classification neural network.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np
from layers.layer import Layer


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
        super().__init__(False)

        # Set the network identification key
        Softmax.layers += 1
        self._network_id = 'Softmax' + str(Softmax.layers)

        # Define the layer calculation retainers
        self._logit_inputs = None
        self._softmax_outputs = None

    def forward(self, logit_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Softmax normalization delineates
        the input values and converts the feature layers into probabilities.
        :param logit_inputs: The inputs to this softmax model layer
        :return: The softmax normalized values of the input along the axis
        """

        # Cache the inputs for backpropagation later
        self._logit_inputs = logit_inputs

        # Use the softmax function to convert logits to probability
        odd_mediums = np.exp(logit_inputs)
        softmax_outputs = odd_mediums / odd_mediums.sum(axis=1)[:, np.newaxis]

        # Cache the outputs for backpropagation later
        self._softmax_outputs = softmax_outputs

        return softmax_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. Because of the multi-input
        nature of the softmax function, softmax backward propagation requires
        indexing a higher dimensional tensor therefore being computationally
        expensive.
        :param output_gradients: The loss derivatives respecting outputs
        :return: The partial derivatives of the loss respecting the inputs
        """

        # Check that forward propagation has been completed
        try:
            layer_inputs = self._logit_inputs[:]
            layer_outputs = self._softmax_outputs[:]
        except TypeError:
            raise ValueError("Forward propagation must be completed.")

        # Define the rank three softmax gradients
        samples = layer_inputs.shape[0]
        logits = layer_inputs.shape[1]
        softmax_gradients = np.zeros((samples, logits, logits))

        # Calculate the partial softmax derivatives respecting each logit
        for max_idx in range(logits):
            for logit_idx in range(max_idx, logits):
                if logit_idx != max_idx:
                    # Make use of the fact the matrices are symmetrical
                    off_gradients = (
                        -layer_outputs[:, max_idx]
                        * layer_outputs[:, logit_idx])
                    softmax_gradients[:, max_idx, logit_idx] = off_gradients
                    softmax_gradients[:, logit_idx, max_idx] = off_gradients
                else:
                    softmax_gradients[:, max_idx, logit_idx] = (
                        layer_outputs[:, logit_idx]
                        * (1 - layer_outputs[:, logit_idx]))

        # Calculate the logit gradients
        logit_gradients = np.squeeze(
            output_gradients[:, np.newaxis]
            @ softmax_gradients)

        return logit_gradients
