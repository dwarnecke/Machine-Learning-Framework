"""
Softmax layer in a multi class classification neural network.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np
from layers.layer import Layer


class Softmax(Layer):
    def __init__(self):
        """
        Create a softmax layer for categorical classification.
        """

        # Call the superclass initializer
        super().__init__(False)

        # Define the layer calculation retainers
        self._logit_inputs = None
        self._softmax_outputs = None

    def initialize(self, units_in: int, layer_idx: int):
        """
        Fully compile the softmax layer by appropriately setting the units
        variable of the layer to connect the model and prepare the layer for
        use.
        :param units_in: The number of units being input to the layer
        :param layer_idx: The layer number in the larger network
        """

        # Check and that layer index and input units are positive integers
        if layer_idx < 1:
            raise ValueError("Layer index must be greater than zero.")
        elif units_in < 1:
            raise ValueError("Input units must be greater than zero.")
        self.features = units_in

        # Set the network identification key
        self._network_id = 'Softmax' + str(layer_idx)

    def forward(self, logit_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Softmax normalization delineates
        the input values and converts the feature layers into probabilities.
        :param logit_inputs: The inputs to this softmax model layer
        :return: The softmax normalized values of the input along the axis
        """

        # Check that the layer is initialized
        if self.features is None:
            raise ValueError("Layer must be initialized first.")

        # Cache the inputs for backpropagation later
        self._logit_inputs = logit_inputs

        # Use the softmax function to convert logits to probability
        odd_mediums = np.exp(logit_inputs)
        softmax_outputs = odd_mediums / odd_mediums.sum(axis=1)[:, np.newaxis]

        # Cache the outputs for backpropagation later
        self._softmax_outputs = softmax_outputs

        return softmax_outputs  # Return the activated softmax values

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
        if self._logit_inputs is None or self._softmax_outputs is None:
            raise ValueError("Forward propagation must be completed first.")

        # Retrieve the forward propagation values
        layer_inputs = self._logit_inputs
        layer_outputs = self._softmax_outputs

        # Define the rank three softmax gradients
        n_samples = layer_inputs.shape[0]
        softmax_gradients = np.zeros((n_samples, self.features, self.features))

        # Calculate the partial softmax derivatives respecting each logit
        for max_idx in range(self.features):
            for logit_idx in range(max_idx, self.features):
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
        logit_gradients = (
                output_gradients[:, np.newaxis]
                @ softmax_gradients).squeeze()

        return logit_gradients  # Return the logit gradients
