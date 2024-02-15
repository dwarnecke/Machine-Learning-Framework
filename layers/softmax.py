"""
Softmax layer in a multi class classification neural network.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np
from layers.layer import Layer


class Softmax(Layer):
    def __init__(self):
        """
        Create a softmax layer for categorical classification.
        """

        super().__init__(False)  # Call the superclass initializer

        self.units = None  # Define the number of units this layer makes

        # Define the layer calculation retainers
        self._logit_inputs = None
        self._softmax_outputs = None

    def compile(self, layer_idx: int, input_units: int):
        """
        Fully compile the softmax layer by appropriately setting the units
        variable of the layer to connect the model and prepare the layer for
        use.
        :param layer_idx: The layer number in the larger network
        :param input_units: The number of units being input to the layer
        """

        # Check and that layer index and input units are positive integers
        if layer_idx < 1:
            raise ValueError("Layer index must be greater than zero.")
        elif input_units < 1:
            raise ValueError("Input units must be greater than zero.")
        self.units = input_units

        # Set the network identification key
        self._network_id = 'Softmax' + str(layer_idx)

        self._is_compiled = True  # Change the layer compilation flag

    def forward(
            self,
            logit_inputs: np.ndarray,
            in_training: bool) -> np.ndarray:
        """
        Pass through this softmax layer in forward propagation. Softmax
        normalization delineates the input values and converts the feature
        layers into probabilities.
        :param logit_inputs: The inputs to this softmax model layer
        :param in_training: If the model is currently being trained
        :return: The softmax normalized values of the input along the axis
        """

        # Check that the layer is initialized
        if not self._is_compiled:
            raise AttributeError("Layer must be compiled.")

        # Check that the input is a two-dimensional numpy array
        if type(logit_inputs) != np.ndarray:
            raise TypeError("Logit inputs must be a numpy array.")
        if np.ndim(logit_inputs) != 2:
            raise ValueError("Logit inputs must be two dimensional.")

        # Check that the input logits matches the number of units
        if logit_inputs.shape[-1] != self.units:
            raise ValueError("Logit inputs size and units must match.")

        # Cache the inputs for later use
        if in_training:
            self._logit_inputs = logit_inputs

        # Use the softmax function to convert logits to probability
        odd_mediums = np.exp(logit_inputs)
        softmax_outputs = odd_mediums / odd_mediums.sum(axis=1)[:, np.newaxis]

        self._softmax_outputs = softmax_outputs  # Cache the outputs for later

        return softmax_outputs  # Return the activated softmax values

    def backward(self, output_grads: np.ndarray) -> np.ndarray:
        """
        Pass through this softmax layer in backward propagation. Because of
        the multi-input nature of the softmax function, softmax backward
        propagation requires indexing a higher dimensional tensor therefore
        being computationally expensive.
        :param output_grads: The loss derivatives respecting outputs
        :return: The partial derivatives of the loss respecting the inputs
        """

        # Check that the layer is initialized
        if not self._is_compiled:
            raise AttributeError("Layer must be compiled.")

        # Check that the input shape is a two-dimensional numpy array
        if type(output_grads) != np.ndarray:
            raise TypeError("Layer inputs must be a numpy array.")
        if np.ndim(output_grads) != 2:
            raise ValueError("Layer inputs must be two dimensional.")

        # Check that forward propagation has been completed
        if self._logit_inputs is None or self._softmax_outputs is None:
            raise ValueError("Forward propagation must be completed first.")

        # Retrieve the forward propagation values
        layer_inputs = self._logit_inputs
        layer_outputs = self._softmax_outputs

        # Define the three-dimensional softmax gradients
        n_samples = layer_inputs.shape[0]
        softmax_grads = np.zeros((n_samples, self.units, self.units))

        # Calculate the partial softmax derivatives respecting each logit
        for max_idx in range(self.units):
            for logit_idx in range(max_idx, self.units):
                if logit_idx != max_idx:
                    # Make use of the fact the matrices are symmetrical
                    off_grads = \
                        -(layer_outputs[:, max_idx]
                          * layer_outputs[:, logit_idx])
                    softmax_grads[:, max_idx, logit_idx] = off_grads
                    softmax_grads[:, logit_idx, max_idx] = off_grads
                else:
                    softmax_grads[:, max_idx, logit_idx] = \
                        (layer_outputs[:, logit_idx]
                         * (1 - layer_outputs[:, logit_idx]))

        # Calculate the logit gradients
        logit_grads = output_grads[:, np.newaxis] @ softmax_grads
        logit_grads = logit_grads.squeeze()

        return logit_grads  # Return the logit gradients
