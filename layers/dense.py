"""
Fully connected, dense layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import math
import numpy as np
from typing import Callable

import activations
from layers.layer import Layer


class Dense(Layer):
    """
    Fully connected, dense layer for a neural network. These layers
    linearly transform, offset, and delineate data to provide a number of
    new, more meaningful features.
    """

    # Define the number of dense layers created
    layers = 0

    def __init__(self, units: int, activation: Callable = None):
        """
        Create a fully connected dense layer for a neural network.
        :param units: The feature dimension of the layer
        :param activation: The activation function of the layer
        """

        # Call the superclass initializer
        super().__init__(True)

        # Check and define the layer output shape
        if activation is not None:
            self._ACTIVATION = activation
        else:
            self._ACTIVATION = activations.linear
        if units < 1:
            raise ValueError("Units must be positive.")
        self.UNITS = units

        # Define the network identification key
        Dense.layers += 1
        self._network_id = 'Dense' + str(Dense.layers)

        # Define the kernel and bias dictionaries
        self.parameters['kernel'] = {}
        self.parameters['bias'] = {}

        # Define the calculation retainers
        self._layer_inputs = None
        self._linear_mediums = None

    def initialize(self, input_shape: tuple) -> None:
        """
        Initialize the parameters to connect the model together and prepare
        the layer for use.
        :param input_shape: The array shape of a single input sample
        """

        # Check that the feature dimension is positive
        if input_shape[-1] < 1:
            raise ValueError("Feature dimension must be positive.")
        input_units = input_shape[-1]

        # Define the layer output shape
        self.OUTPUT_SHAPE = input_shape[:-1] + (self.UNITS,)

        # Define the parameter identification keys
        self.parameters['kernel']['id'] = self._network_id + '_kernel'
        self.parameters['bias']['id'] = self._network_id + '_bias'

        # Initialize the kernel according to activation function
        generator = np.random.default_rng()
        size = (input_shape[-1], self.UNITS)
        if self._ACTIVATION in (activations.sigmoid, activations.tanh):
            # Use Xavier initialization
            scale = math.sqrt(6 / (input_units + self.UNITS))
            kernel = generator.uniform(-scale, scale, size)
        elif self._ACTIVATION == activations.relu:
            # Use He initialization
            scale = math.sqrt(2 / input_units)
            kernel = generator.normal(0, scale, size)
        else:
            scale = math.sqrt(1 / input_units)
            kernel = generator.normal(0, scale, size)
        self.parameters['kernel']['values'] = kernel

        # Initialize the bias at zero
        size = len(input_shape) * (1,) + (self.UNITS,)
        self.parameters['bias']['values'] = np.zeros(size)

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Inputs are linearly transformed,
        offset by a bias, and delineated with a nonlinear activation function.
        :param layer_inputs: The inputs to this dense layer
        :return: The activated values that the layer calculates
        """

        # Check that the parameter values are initialized
        try:
            kernel = self.parameters['kernel']['values']
            bias = self.parameters['bias']['values']
        except KeyError:
            raise ValueError("Parameter values must be initialized.")

        # Transform and delineate the layer inputs
        activation = self._ACTIVATION
        linear_mediums = layer_inputs @ kernel + bias
        layer_outputs = activation(linear_mediums)

        # Save the calculations for backpropagation later
        self._layer_inputs = layer_inputs
        self._linear_mediums = linear_mediums

        return layer_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Backward propagate through this layer. Gradients
        will propagate in proportion to the weights associated and the
        derivative of the activation function.
        :param output_gradients: The loss gradients respecting layer outputs
        :return: The partial derivatives of the loss with respect of inputs
        """

        # Check that forward propagation has been completed
        try:
            linear_mediums = self._linear_mediums[:]  # None cannot be indexed
            layer_inputs = self._layer_inputs[:]
        except TypeError:
            raise ValueError("Forward propagation must be completed.")

        # Retrieve the remaining layer attributes
        activation = self._ACTIVATION
        kernel = self.parameters['kernel']['values']

        # Calculate the input gradients
        activation_gradients = activation(linear_mediums, differentiate=True)
        linear_gradients = activation_gradients * output_gradients
        input_gradients = np.dot(linear_gradients, kernel.T)

        # Calculate the parameter gradients
        kernel_gradients = np.dot(layer_inputs.T, linear_gradients)
        bias_gradients = linear_gradients.sum(axis=0, keepdims=True)

        # Save the parameter gradients for updating later
        self.parameters['kernel']['gradients'] = kernel_gradients
        self.parameters['bias']['gradients'] = bias_gradients

        # Clear the retainers to prevent double use
        self._linear_mediums = None
        self._layer_inputs = None

        return input_gradients
