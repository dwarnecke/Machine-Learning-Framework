"""
Fully connected, dense layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

import math
import numpy as np
from numpy import ndarray
from framework import activations
from framework.activations import Activation
from framework.activations import Linear
from framework.layers import serialize_parameters
from framework.layers.layer import Layer


class Dense(Layer):
    """
    Fully connected, dense layer for a neural network. These layers
    linearly transform, offset, and delineate data to provide a number of
    new, more meaningful features.
    """

    # Define the number of dense layers created
    layers = 0

    def __init__(self, units: int, activation: Activation = None):
        """
        Create a fully connected, dense layer for a neural network.
        :param units: The feature dimension of the layer
        :param activation: The activation function of the layer
        """

        # Call the superclass initializer
        super().__init__()

        # Define the layer identification key
        Dense.layers += 1
        self.LAYER_ID = 'Dense' + str(Dense.layers)

        # Check that the number of feature units is positive
        if units < 1:
            raise ValueError("Units must be positive.")
        else:
            self._UNITS = units

        # Define the default activation as the linear function
        if activation is None:
            self._ACTIVATION = Linear()
        else:
            self._ACTIVATION = activation

        # Define the parameter dictionaries and identification keys
        self.parameters = {'kernel': {}, 'bias': {}}
        self.parameters['kernel']['id'] = self.LAYER_ID + '_kernel'
        self.parameters['bias']['id'] = self.LAYER_ID + '_bias'

        # Define the forward propagation retainers and flags
        self._input_layers = None
        self._linear_mediums = None
        self._is_configured = False

    def _configure(self, inputs_shape: tuple) -> bool:
        """
        Configure the layer to be used.
        :param inputs_shape: The shape of the layer input data
        :return: If the layer was successfully configured
        """

        # Rename different layer dimensions for ease
        units_in = inputs_shape[-1]
        units_out = self._UNITS

        # Initialize the kernel according to activation function
        generator = np.random.default_rng()
        size = (units_in, units_out)
        if isinstance(self._ACTIVATION, activations.xavier_activations):
            # Use Xavier initialization
            scale = math.sqrt(6 / (units_in + units_out))
            kernel = generator.uniform(-scale, scale, size)
        elif isinstance(self._ACTIVATION, activations.he_activations):
            # Use He initialization
            scale = math.sqrt(2 / units_in)
            kernel = generator.normal(0, scale, size)
        else:
            scale = math.sqrt(1 / units_in)
            kernel = generator.normal(0, scale, size)
        self.parameters['kernel']['values'] = kernel

        # Initialize the bias at zero
        size = len(inputs_shape) * (1,) + (units_out,)
        self.parameters['bias']['values'] = np.zeros(size)

        return True

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to this dense layer
        :return: The activated values that the layer calculates
        """

        # Check that the layer is configured
        if not self._is_configured:
            inputs_shape = input_layers.shape[1:]
            self._is_configured = self._configure(inputs_shape)

        # Transform and delineate the layer inputs
        kernel = self.parameters['kernel']['values']
        bias = self.parameters['bias']['values']
        linear_mediums = input_layers @ kernel + bias
        output_layers = self._ACTIVATION.calculate(linear_mediums)

        # Save the calculations for backpropagation later
        self._input_layers = input_layers
        self._linear_mediums = linear_mediums

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss gradients respecting the layer outputs
        :return: The partial loss derivatives with respect to layer inputs
        """

        # Check that forward propagation has been completed
        input_layers = self._input_layers[:]  # None cannot be indexed
        linear_mediums = self._linear_mediums[:]
        self._input_layers = None  # Erase the retainers to prevent repeat use
        self._linear_mediums = None

        # Adjust the gradients by the activation function
        activation_grads = self._ACTIVATION.differentiate(linear_mediums)
        linear_grads = activation_grads * output_grads

        # Calculate the parameter and input gradients
        kernel = self.parameters['kernel']['values']
        kernel_grads = np.dot(input_layers.T, linear_grads)
        bias_grads = linear_grads.sum(axis=0, keepdims=True)
        input_grads = np.dot(linear_grads, kernel.T)

        # Save the parameter gradients for updating later
        self.parameters['kernel']['grads'] = kernel_grads
        self.parameters['bias']['grads'] = bias_grads

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        # Check that the layer is configured
        if not self._is_configured:
            raise ValueError("Layer must be configured before serializing.")

        # Serialize the needed attributes
        activation = self._ACTIVATION.serialize()
        parameters = serialize_parameters(self.parameters)

        # Collect all the layer attributes
        layer = {
            'type': 'dense',
            'units': self._UNITS,
            'activation': activation,
            'parameters': parameters
        }

        return layer