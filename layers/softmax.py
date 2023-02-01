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

        self.UNITS = None  # Define the number of units this layer makes

        # Define the layer calculation retainers
        self._logit_inputs = None
        self._softmax_outputs = None

    def compile(self, input_units: int):
        """
        Fully compile the softmax layer by appropriately setting the units
        variable of the layer.
        :param input_units: The number of logit units being input to the layer
        """

        # Check and properly initialize the units in this layer
        if type(input_units) != int:
            raise TypeError("Input units must be an integer.")
        elif input_units < 1:
            raise ValueError("Input units must be greater than zero.")
        self.UNITS = input_units

        self._is_compiled = True  # Change the layer compilation flag

    def forward(self, logit_inputs: np.ndarray) -> np.ndarray:
        """
        Pass through this softmax layer with forward propagation.
        :param logit_inputs: The inputs to this softmax model layer
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
        if logit_inputs.shape[1] != self.UNITS:
            raise ValueError("Input axis dimension size and units must match.")

        self._logit_inputs = logit_inputs  # Cache the inputs for later use

        # Use the softmax function to convert logits to probability
        odd_mediums = np.exp(logit_inputs)
        softmax_outputs = odd_mediums / odd_mediums.sum(axis=1)[:, np.newaxis]

        self._softmax_outputs = softmax_outputs  # Cache the outputs for later

        return softmax_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Pass through this layer with backward propagation by calculating the
        partial derivatives of the loss function with respect to the softmax
        input logits.
        :param output_gradients: The loss derivatives respecting outputs
        :return: The partial derivatives of the loss respecting the inputs
        """

        # Check that the layer is initialized
        if not self._is_compiled:
            raise AttributeError("Layer must be compiled.")

        # Check that the input shape is a two-dimensional numpy array
        if type(output_gradients) != np.ndarray:
            raise TypeError("Layer inputs must be a numpy array.")
        if np.ndim(output_gradients) != 2:
            raise ValueError("Layer inputs must be two dimensional.")

        # Check that forward propagation has been completed
        if self._logit_inputs is None or self._softmax_outputs is None:
            raise ValueError("Forward propagation must be completed first.")

        # Retrieve the forward propagation values
        layer_inputs = self._logit_inputs
        layer_outputs = self._softmax_outputs

        # Define the three-dimensional softmax gradients
        n_samples = layer_inputs.shape[0]
        softmax_gradients = np.zeros((n_samples, self.UNITS, self.UNITS))

        # Calculate the partial softmax derivatives respecting each logit
        for softmax_idx in range(self.UNITS):
            for logit_idx in range(softmax_idx, self.UNITS):
                if logit_idx != softmax_idx:
                    # Make use of the symmetrical matrix property
                    off_gradient = \
                        -layer_outputs[:, softmax_idx] \
                        * layer_outputs[:, logit_idx]
                    softmax_gradients[:, softmax_idx, logit_idx] = off_gradient
                    softmax_gradients[:, logit_idx, softmax_idx] = off_gradient
                else:
                    softmax_gradients[:, softmax_idx, logit_idx] = \
                        layer_outputs[:, logit_idx] \
                        * (1 - layer_outputs[:, logit_idx])

        # Calculate the logit gradients
        logit_gradients = output_gradients[:, np.newaxis] @ softmax_gradients
        logit_gradients = logit_gradients.squeeze()

        return logit_gradients
