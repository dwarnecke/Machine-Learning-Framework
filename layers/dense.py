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
    def __init__(self, units: int, activation=None):
        """
        Create a fully connected dense layer for a neural network.
        :param units: The number of nodes the layer leads to
        :param str activation: The activation function of the layer
        """

        # Call the superclass initializer
        super().__init__(True)

        # Check and define the layer units
        if type(units) != int:
            raise ValueError("Units must be integers.")
        if units < 1:
            raise ValueError("Units must be greater than zero.")
        self.UNITS = units

        # Check and define the layer activation function
        if not activations.verify_activation(activation):
            raise TypeError("Activation must be valid functions.")
        self._ACTIVATION = activation

        # Define the layer weights and biases
        self._weight = None
        self._bias = None

        # Define the layer calculation and parameter gradient retainers
        self._layer_inputs = None
        self._linear_mediums = None
        self._weight_gradients = None
        self._bias_gradients = None

        # Define the network identification and parameter keys
        self.network_id = None
        self._weight_id = None
        self._bias_id = None

    def compile(self, layer_idx: int, input_units: int):
        """
        Initialize the parameters to connect the model together and prepare
        the layer for use.
        :param layer_idx: The layer number in the larger network
        :param input_units: The number of nodes leading into this layer
        """

        # Check and that layer index and input units are positive integers
        if layer_idx < 1:
            raise ValueError("Layer index must be greater than zero.")
        elif input_units < 1:
            raise ValueError("Input units must be greater than zero.")

        # Set the network identification and parameter keys
        self.network_id = 'Dense' + str(layer_idx)
        self._weight_id = self.network_id + '_weight'
        self._bias_id = self.network_id + '_bias'

        # Initialize the weights according to activation function
        generator = np.random.default_rng()
        if self._ACTIVATION in ('sigmoid', 'tanh'):
            # Initialize the weight using Xavier initialization
            self._weight = generator.uniform(
                -6 / math.sqrt(input_units + self.UNITS),
                6 / math.sqrt(input_units + self.UNITS),
                (input_units, self.UNITS))
        elif self._ACTIVATION == 'relu':
            # Initialize weights using He initialization
            self._weight = generator.normal(
                0, 2 / input_units, (input_units, self.UNITS))
        elif self._ACTIVATION is None:
            self._weight = generator.normal(
                0, 1 / input_units, (input_units, self.UNITS))

        self._bias = np.zeros((1, self.UNITS))  # Initialize the bias at zero

        self._is_compiled = True  # Change the layer compilation flag

    def forward(
            self,
            layer_inputs: np.ndarray,
            in_training: bool) -> np.ndarray:
        """
        Pass through this dense layer in forward propagation. Inputs are
        linearly transformed, offset by a bias, and delineated with a
        nonlinear activation function.
        :param layer_inputs: The inputs to this dense layer
        :param in_training: If the model is currently training
        :return: The activated values that the layer calculates
        """

        # Check that the parameters are initialized
        if not self._is_compiled:
            raise AttributeError("Parameters must be initialized.")

        # Check that the input shape is a two-dimensional numpy array
        if type(layer_inputs) != np.ndarray:
            raise TypeError("Layer inputs must be a numpy array.")
        if np.ndim(layer_inputs) != 2:
            raise ValueError("Layer inputs must be two dimensional.")

        # Check that the input layer shape matches the weight shape
        if layer_inputs.shape[1] != self._weight.shape[0]:
            raise ValueError("Weight and input middle dimension must match.")

        # Scale by the weights and offset with bias
        linear_mediums = np.dot(layer_inputs, self._weight) + self._bias

        # Delineate the outputs using activation functions
        layer_outputs = activations.calculate(
            self._ACTIVATION,
            linear_mediums)

        if in_training:
            # Cache these values for backpropagation later
            self._layer_inputs = layer_inputs
            self._linear_mediums = linear_mediums

        return layer_outputs  # Return the propagated inputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Pass through this dense layer in backward propagation. Gradients
        will propagate in proportion to the weights associated and the
        derivative of the activation function.
        :param output_gradients: The loss gradients respecting outputs
        :return: The partial derivatives of the loss with input respect
        """

        # Check that the parameters are initialized
        if not self._is_compiled:
            raise AttributeError("Parameters must be initialized.")

        # Check that the input is a two-dimensional numpy array
        if type(output_gradients) != np.ndarray:
            raise TypeError("Output gradients must be a numpy array.")
        if np.ndim(output_gradients) != 2:
            raise ValueError("Output gradients must be two dimensional.")

        # Check that forward propagation has been completed
        if self._linear_mediums is None or self._layer_inputs is None:
            raise ValueError("Forward propagation must be completed first.")

        # Check that the output layer shape matches the number of layer units
        if output_gradients.shape[1] != self.UNITS:
            raise ValueError("Output and weight middle dimension must match.")

        # Retrieve the forward propagation values
        linear_mediums = self._linear_mediums
        layer_inputs = self._layer_inputs

        # Calculate the activation derivatives and linear gradients
        activation_gradients = activations.differentiate(
            self._ACTIVATION,
            linear_mediums)
        linear_gradients = activation_gradients * output_gradients

        # Calculate the weight and layer gradients
        weight_gradients = np.dot(layer_inputs.T, linear_gradients)
        bias_gradients = linear_gradients.sum(axis=0)[np.newaxis]
        input_gradients = np.dot(linear_gradients, self._weight.T)

        # Save the parameter gradients for updating later
        self._weight_gradients = weight_gradients
        self._bias_gradients = bias_gradients

        return input_gradients  # Return the layer gradients

    def update(
            self,
            optimizer: optimizers.valid_optimizers,
            learning_rate: float):
        """
        Update the weight and bias parameters in one step of gradient descent.
        :param optimizer: The optimizer to adjust the parameters with
        :param learning_rate: The rate at which to change the parameters by
        """

        # Check that the learning rate is a positive float
        if not(type(learning_rate) in (float, int)):
            raise TypeError("Learning rate must be a float or an integer.")
        elif learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        # Check that the parameters are initialized
        if not self._is_compiled:
            raise AttributeError("Parameters must be initialized.")

        # Calculate the parameter adjustment
        if optimizer is not None:
            weight_delta = optimizer.calculate_adjustment(
                self._weight_id, self._weight_gradients, learning_rate)
            bias_delta = optimizer.calculate_adjustment(
                self._bias_id, self._bias_gradients, learning_rate)
        else:
            weight_delta = learning_rate * self._weight_gradients
            bias_delta = learning_rate * self._bias_gradients

        # Update the parameters with gradient descent
        self._weight = self._weight - weight_delta
        self._bias = self._bias - bias_delta
