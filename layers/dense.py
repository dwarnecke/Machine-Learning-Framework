"""
Dense layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import math
import numpy as np

import activations
import optimizers
from layers.layer import Layer


class Dense(Layer):
    """
    Fully connected, dense layer for a neural network. These layers
    linearly transform, offset, and delineate data to provide a number of
    new, more meaningful features.
    """

    def __init__(
            self,
            units: int,
            activation: activations.valid_activations = None):
        """
        Create a fully connected dense layer for a neural network.
        :param units: The feature dimension of the layer
        :param activation: The activation function of the layer
        """

        # Call the superclass initializer
        super().__init__(True)

        # Check and define the layer architecture parameters
        if activation is not None:
            self._ACTIVATION = activation
        else:
            self._ACTIVATION = activations.linear
        if units < 1:
            raise ValueError("Units must be positive.")
        self.features = units

        # Define the kernel and bias dictionaries
        self._kernel = {}
        self._bias = {}

        # Define the calculation retainers
        self._layer_inputs = None
        self._linear_mediums = None

    def initialize(self, units_in: int, layer_idx: int) -> None:
        """
        Initialize the parameters to connect the model together and prepare
        the layer for use.
        :param units_in: The feature dimension leading into this layer
        :param layer_idx: The layer number in the larger network
        """

        # Check and that passed integer arguments are positive
        if units_in < 1:
            raise ValueError("Number of input units must be positive.")
        elif layer_idx < 1:
            raise ValueError("Layer index must be positive.")

        # Define the network identification and parameter keys
        self._network_id = 'Dense' + str(layer_idx)
        self._kernel['id'] = self._network_id + '_kernel'
        self._bias['id'] = self._network_id + '_bias'

        # Initialize the kernel according to activation function
        generator = np.random.default_rng()
        size = (units_in, self.features)
        if self._ACTIVATION in (activations.sigmoid, activations.tanh):
            # Use Xavier initialization
            scale = math.sqrt(6 / (units_in + self.features))
            self._kernel['values'] = generator.uniform(-scale, scale, size)
        elif self._ACTIVATION == activations.relu:
            # Use He initialization
            scale = math.sqrt(2 / units_in)
            self._kernel['values'] = generator.normal(0, scale, size)
        elif self._ACTIVATION == activations.linear:
            scale = math.sqrt(1 / units_in)
            self._kernel['values'] = generator.normal(0, scale, size)

        # Initialize the bias at zero
        self._bias['values'] = np.zeros((1, self.features))

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Inputs are linearly transformed,
        offset by a bias, and delineated with a nonlinear activation function.
        :param layer_inputs: The inputs to this dense layer
        :return: The activated values that the layer calculates
        """

        # Check that the layer is initialized
        try:
            kernel = self._kernel['values']
            bias = self._bias['values']
        except KeyError:
            raise KeyError("Parameter values must be initialized first.")

        # Check that the input layer shape matches the kernel shape
        if layer_inputs.shape[-1] != self._kernel['values'].shape[0]:
            raise ValueError("Kernel and inputs dimension must match.")

        # Calculate the layer outputs by transforming and delineating
        activation = self._ACTIVATION
        linear_mediums = layer_inputs @ kernel + bias  # Transform
        layer_outputs = activation(linear_mediums)  # Delineate

        # Cache these values for backpropagation later
        self._layer_inputs = layer_inputs
        self._linear_mediums = linear_mediums

        return layer_outputs  # Return the propagated inputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. Gradients
        will propagate in proportion to the weights associated and the
        derivative of the activation function.
        :param output_gradients: The loss gradients respecting layer outputs
        :return: The partial derivatives of the loss with respect of inputs
        """

        # Check that forward propagation has been completed
        if self._linear_mediums is None or self._layer_inputs is None:
            raise KeyError("Forward propagation must be completed first.")

        # Check that the output layer shape matches the number of layer units
        if output_gradients.shape[1] != self.features:
            raise ValueError("Gradients and kernel dimension must match.")

        # Calculate the input gradients
        activation = self._ACTIVATION
        linear_mediums = self._linear_mediums
        kernel_values = self._kernel['values']
        activation_gradients = activation(linear_mediums, differentiate=True)
        linear_gradients = activation_gradients * output_gradients
        input_gradients = np.dot(linear_gradients, kernel_values.T)

        # Calculate the parameter gradients
        layer_inputs = self._layer_inputs
        kernel_gradients = np.dot(layer_inputs.T, linear_gradients)
        bias_gradients = linear_gradients.sum(axis=0)[np.newaxis]

        # Cache the parameter gradients for updating later
        self._kernel['gradients'] = kernel_gradients
        self._bias['gradients'] = bias_gradients

        return input_gradients  # Return the layer input gradients

    def update(
            self,
            optimizer: optimizers.valid_optimizers,
            learning_rate: int or float) -> None:
        """
        Update the weight and bias parameters in one step of gradient descent.
        :param optimizer: The optimizer to adjust the parameters with
        :param learning_rate: The rate at which to change the parameters by
        """

        # Check that the learning rate is positive
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        # Check that backward propagation has been completed
        try:
            self._kernel['gradients'] = self._kernel['gradients']
            self._bias['gradients'] = self._bias['gradients']
        except KeyError:
            raise KeyError("Backward propagation must be completed first.")

        # Calculate the gradient descent parameter deltas
        kernel_delta = learning_rate * self._kernel['gradients']
        bias_delta = learning_rate * self._bias['gradients']

        # Calculate the optimizer parameter deltas if given
        if optimizer is not None:
            kernel_delta = optimizer.calculate_delta(
                self._kernel['id'],
                self._kernel['gradients'],
                learning_rate)
            bias_delta = optimizer.calculate_delta(
                self._bias['id'],
                self._bias['gradients'],
                learning_rate)

        # Update the parameters by their deltas
        self._kernel['values'] = self._kernel['values'] - kernel_delta
        self._bias['values'] = self._bias['values'] - bias_delta
