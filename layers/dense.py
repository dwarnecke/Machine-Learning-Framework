"""
Dense, fully connected layer in a machine learning model
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import math
import numpy as np
from layers.layer import Layer
import activations


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

    def compile(self, units: int):
        """
        Initialize parameters to connect the layer properly in a model.
        :param units: The number of nodes leading into this layer
        """

        # Check and that input units is a positive integer
        if type(units) != int:
            raise TypeError("Input units must be an integer.")
        elif units < 1:
            raise ValueError("Input units must be greater than zero.")

        # Initialize the weights according to activation function
        generator = np.random.default_rng()
        if self._ACTIVATION in ('sigmoid', 'tanh'):
            # Initialize the weight using Xavier initialization
            self._weight = generator.uniform(
                -6 / math.sqrt(units + self.UNITS),
                6 / math.sqrt(units + self.UNITS),
                (units, self.UNITS))
        elif self._ACTIVATION == 'relu':
            # Initialize weights using He initialization
            self._weight = generator.normal(0, 2 / units, (units, self.UNITS))
        elif self._ACTIVATION is None:
            self._weight = generator.normal(0, 1 / units, (units, self.UNITS))

        # Initialize the bias at zero
        self._bias = np.zeros((1, self.UNITS))

        # Change the layer compilation flag
        self._is_compiled = True

    def forward(self, layer_inputs: np.ndarray) -> np.ndarray:
        """
        Pass through this dense layer with forward propagation.
        :param layer_inputs: The inputs to this dense layer
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
        layer_outputs = activations.calculate(self._ACTIVATION, linear_mediums)

        # Cache these values for backpropagation later
        self._layer_inputs = layer_inputs
        self._linear_mediums = linear_mediums

        return layer_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """
        Pass through this dense layer with backward propagation by finding
        :param output_gradients: The loss derivatives respecting outputs
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

        return input_gradients

    def update(self, learning_rate: float or int):
        """
        Update the weight and bias parameters in one step of gradient descent.
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

        # Update the parameters with gradient descent
        self._weight = self._weight - learning_rate * self._weight_gradients
        self._bias = self._bias - learning_rate * self._bias_gradients
