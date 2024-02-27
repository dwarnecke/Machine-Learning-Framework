"""
Two-dimensional convolution layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import math
import numpy as np

from layers.layer import Layer


def _pad(inputs: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Pad zeros around images to prevent loss in convolution.
    :param inputs: The input data to be padded
    :param padding_size: The number of zeros to pad the image with
    :return: The padded data of the images
    """

    # Check that the padding size is positive
    if padding_size < 1:
        raise ValueError("Padding size must be positive.")

    # Define the padded images as a default array of zeroes
    padded_images_shape = list(inputs.shape)
    padded_images_shape[1] += 2 * padding_size
    padded_images_shape[2] += 2 * padding_size
    padded_images = np.zeros(padded_images_shape)

    # Replace the intermediate zeroes with the original data
    padded_images[
        :,
        padding_size:(inputs.shape[1] + padding_size),
        padding_size:(inputs.shape[2] + padding_size)] = inputs

    return padded_images


def _convolve(inputs: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """
    Convolve many input channels and kernels.
    :param inputs: The inputs to use in the convolutions
    :param kernels: The kernel to use in the convolutions
    :return: The convolution feature maps of the inputs and kernels
    """

    # Check that the input size is larger than the kernel
    kernel_size = kernels.shape[0]
    input_height = inputs.shape[1]
    input_width = inputs.shape[2]
    if input_height < kernel_size or input_width < kernel_size:
        raise IndexError("Inputs must be larger than the kernel.")

    # Calculate convolutions of all the kernels
    feature_maps = np.array(
        [   # For every window row of the inputs
            [   # For every window column of the inputs
            np.tensordot(
                inputs[:, x:(x + kernel_size), y:(y + kernel_size)],
                kernels,
                (3, 2, 1))
            for y in range(input_width - kernel_size + 1)]
        for x in range(input_height - kernel_size + 1)]
    ).transpose((2, 0, 1, 3))  # Example, row, column, channel indexing

    return feature_maps


class Convolution(Layer):
    """
    Two-dimensional convolution layer for a neural network. These layers
    convolve windows of the layer input with kernels of the same size and
    offset those new features with the respective bias of the kernel.
    """

    # Define the number of convolution layer created
    layers = 0

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            padding: bool = False) -> None:
        """
        Create a two-dimensional convolution layer for a neural network
        :param channels: The number of output filters in the convolution
        :param kernel_size: The size of each of the convolution windows
        :param bool padding: If zero padding is added before convolution
        """

        # Call the superclass initializer
        super().__init__(True)

        # Check and define the layer hyperparameters
        if channels < 1:
            raise ValueError("Number of channels must be positive.")
        elif kernel_size < 1:
            raise ValueError("Kernel size must be positive.")
        elif kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd.")
        self._CHANNELS = channels
        self._KERNEL_SIZE = kernel_size
        self._PADDING = padding

        # Define the network identification key
        Convolution.layers += 1
        self._network_id = 'Dense' + str(Convolution.layers)

        # Define the kernels and biases dictionaries
        self.parameters['kernel'] = {}
        self.parameters['bias'] = {}

        # Define the calculation retainers
        self._layer_inputs = None

    def initialize(self, input_shape: tuple) -> None:
        """
        Initialize the parameters to prepare the layer for use.
        :param input_shape: The array shape of a single input sample
        """

        # Check that the feature dimension is positive
        if input_shape[-1] < 1:
            raise ValueError("Feature dimension must be positive.")
        input_channels = input_shape[-1]

        # Define the layer output shape
        if self._PADDING:
            self.OUTPUT_SHAPE = input_shape[:-1] + (self._CHANNELS,)
        else:
            paring_size = self._KERNEL_SIZE - 1
            output_shape = list(input_shape[:-1] + (self._CHANNELS,))
            output_shape[-3] -= paring_size
            output_shape[-2] -= paring_size
            self.OUTPUT_SHAPE = tuple(output_shape)

        # Define the parameter identification keys
        self.parameters['kernels']['id'] = self._network_id + '_kernels'
        self.parameters['biases']['id'] = self._network_id + '_biases'

        # Initialize the kernels using Xavier initialization
        generator = np.random.default_rng()
        fan_in = input_channels * self._KERNEL_SIZE ** 2
        fan_out = self._CHANNELS
        scale = math.sqrt(2 / (fan_in + fan_out))
        size = (self._CHANNELS,) + 2 * (self._KERNEL_SIZE,) + (input_channels,)
        kernels = generator.normal(0, scale, size)
        self.parameters['kernels']['values'] = kernels

        # Initialize the biases at zero
        size = len(input_shape) * (1,) + (self._CHANNELS,)
        biases = np.zeros(size)
        self.parameters['biases']['values'] = biases

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Inputs are convolved by the
        kernels and offset by the biases.
        :param layer_inputs: The inputs to this convolution layer
        :return: The outputs given the inputs and current parameters
        """

        # Check that the parameters are initialized
        try:
            kernels = self.parameters['kernels']['values']
            biases = self.parameters['biases']['values']
        except KeyError:
            raise ValueError("Parameter values must be initialized.")

        # Cache these inputs for backpropagation later
        self._layer_inputs = layer_inputs

        # Pad the inputs adequately
        padded_inputs = layer_inputs
        if self._PADDING:
            paring_size = int((self._KERNEL_SIZE - 1) / 2)
            padded_inputs = _pad(layer_inputs, paring_size)

        # Convolve the inputs and offset by the bias
        layer_outputs = _convolve(padded_inputs, kernels) + biases

        return layer_outputs

    def backward(self, output_gradients) -> np.ndarray:
        """
        Backward propagate through this layer. Gradients
        will be propagated by a transpose convolution with the kernels.
        :param output_gradients: The loss gradients respecting layer outputs
        :return: The partial derivatives of the loss with respect to inputs
        """

        # Check that forward propagation has been completed
        try:
            layer_inputs = self._layer_inputs[:]
        except TypeError:
            raise ValueError("Forward propagation must be completed.")

        # Pad the input values adequately for calculating kernel gradients
        padded_inputs = layer_inputs
        if self._PADDING:
            padding_size = int((self._KERNEL_SIZE - 1) / 2)
            padded_inputs = _pad(layer_inputs, padding_size)

        # Pad the output gradients adequately for calculating input gradients
        padding_size = self._KERNEL_SIZE - 1
        padded_gradients = _pad(output_gradients, padding_size)
        if self._PADDING:
            padding_size = int(padding_size / 2)
            padded_gradients = _pad(output_gradients, padding_size)

        # Calculate the kernels gradients
        kernels_gradients = _convolve(padded_inputs, output_gradients)
        self.parameters['kernels']['gradients'] = kernels_gradients

        # Calculate the biases gradients
        biases_gradients = output_gradients.sum(axis=(0, 1, 2), keepdims=True)
        self.parameters['biases']['gradients'] = biases_gradients

        # Calculate the input gradients
        kernels = self.parameters['kernels']['values']
        flipped_kernels = kernels[:, ::-1, ::-1]
        input_gradients = _convolve(padded_gradients, flipped_kernels)

        return input_gradients

