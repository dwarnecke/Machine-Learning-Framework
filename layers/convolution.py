"""
Two dimensional convolution layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import math
import numpy as np

from layers.layer import Layer


def pad(images: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Pad zeros around images to prevent loss in convolution.
    :param images: The images to be padded
    :param padding_size: The number of zeros to pad the image with
    :return: The padded data of the images
    """

    # Check that the images have appropriate rank
    if np.ndim(images) != 4:
        raise IndexError("Images must have rank four.")

    # Check that the padding size is positive
    if padding_size < 1:
        raise ValueError("Padding size must be positive.")

    # Define the padded images as a default array of zeroes
    padded_images_shape = list(images.shape)
    padded_images_shape[1] += 2 * padding_size
    padded_images_shape[2] += 2 * padding_size
    padded_images = np.zeros(padded_images_shape)

    # Replace the intermediate zeroes with the original data
    padded_images[
        :,
        padding_size:(images.shape[1] + padding_size),
        padding_size:(images.shape[2] + padding_size)] = images

    return padded_images  # Return the newly padded images


def convolve(inputs: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """
    Convolve many input channels and kernels.
    :param inputs: The inputs to use in the convolutions
    :param kernels: The kernel to use in the convolutions
    :return: The convolution feature maps of the inputs and kernels
    """

    # Check that inputs and kernels have appropriate rank
    if np.ndim(inputs) != 4:
        raise IndexError("Images must have rank four.")
    elif np.ndim(kernels) != 4:
        raise IndexError("Kernels must have rank four.")

    # Check that the input size is larger than the kernel
    kernel_size = kernels.shape[0]
    input_height = inputs.shape[1]
    input_width = inputs.shape[2]
    if input_height < kernel_size or input_width < kernel_size:
        raise IndexError("Inputs must be larger than the kernel.")

    # Check that input channel dimension of the kernels and inputs match
    if inputs.shape[-1] != kernels.shape[-1]:
        raise IndexError("Number of input and kernel channels must match.")

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

    return feature_maps  # Return the convolution feature maps


def transpose_convolve(
        gradients: np.ndarray,
        kernels: np.ndarray) -> np.ndarray:
    """
    Transpose convolve many gradients and kernels.
    :param gradients: The gradients to use in the transposed convolution
    :param kernels: The kernels to use in the transposed convolution
    :return: The original input channel gradients
    """

    # Check that the gradients and kernels have appropriate rank
    if np.ndim(gradients) != 4:
        raise IndexError("Gradients must have rank four.")
    elif np.ndim(kernels) != 4:
        raise IndexError("Kernels must have rank four.")

    # Check that the channel dimension of the gradients and kernels are equal
    if gradients.shape[-1] != kernels.shape[-1]:
        raise IndexError("Number of gradient and kernel channels must match.")

    # Pad the gradients with zeroes
    padding_size = kernels.shape[1] - 1
    padded_gradients = pad(gradients, padding_size)

    # Convolve the padded gradients with the kernels
    input_gradients = convolve(padded_gradients, kernels)

    return input_gradients  # Return the transpose convolved gradients


class Convolution(Layer):
    """
    Two-dimensional convolution layer for a neural network. These layers
    convolve windows of the layer input with kernels of the same size and
    offset those new features with the respective bias of the kernel.
    """

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            padding: bool=False) -> None:
        """
        Create a two-dimensional convolution layer for a neural network
        :param channels: The number of output filters in the convolution
        :param kernel_size: The size of each of the convolution windows
        :param bool padding: If zero padding is added before convolution
        """

        # Call the superclass initializer
        super().__init__(True)

        # Check and define the layer architecture parameters
        if channels < 1:
            raise ValueError("Number of channels must be positive.")
        elif kernel_size < 1:
            raise ValueError("Kernel size must be positive.")
        elif kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd.")
        self._CHANNELS = channels
        self._KERNEL_SIZE = kernel_size
        self._PADDING = padding

        # Define the kernels and biases dictionaries
        self._kernels = {}
        self._biases = {}

        # Define the calculation retainers
        self._layer_inputs = None

    def initialize(self, channels_in: int, layer_idx: int):
        """
        Initialize the parameters to prepare the layer for use.
        :param channels_in: The number of channels leading into the layer
        :param layer_idx: The layer number in the larger model
        """

        # Check that the passed integer arguments are positive
        if channels_in < 1:
            raise ValueError("Number of input channels must be positive.")
        elif layer_idx < 1:
            raise ValueError("Layer index must be positive.")

        # Define the network identification and parameter keys
        self._network_id = 'Dense' + str(layer_idx)
        self._kernels['id'] = self._network_id + '_kernels'
        self._biases['id'] = self._network_id + '_biases'

        # Initialize the kernels using Xavier initialization
        generator = np.random.default_rng()
        fan_in = channels_in * self._KERNEL_SIZE ** 2
        fan_out = self._CHANNELS
        scale = math.sqrt(2 / (fan_in + fan_out))
        size = (self._CHANNELS,) + 2 * (self._KERNEL_SIZE,) + (channels_in,)
        self._kernels['values'] = generator.normal(0, scale, size)

        # Initialize the biases at zero
        self._biases = np.zeros((1, 1, self._CHANNELS))

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward propagate through this layer. Inputs are convolved by the
        kernels and offset by the biases.
        :param layer_inputs: The inputs to this convolution layer
        :return: The outputs given the inputs and current parameters
        """

        # Check that the parameters are initialized
        if self._kernels['values'] is None or self._biases['values'] is None:
            raise AttributeError("Parameters must be initialized.")

        # Cache the inputs for backpropagation later
        self._layer_inputs = layer_inputs

        # Pad the layer inputs if so desired
        if self._PADDING:
            paring_size = int((self._KERNEL_SIZE - 1) / 2)
            padded_inputs = pad(layer_inputs, paring_size)
        else:
            padded_inputs = layer_inputs

        # Convolve the inputs and offset by the bias
        layer_outputs = (
                convolve(padded_inputs, self._kernels['values'])
                + self._biases['values'])

        return layer_outputs  # Return the convolved outputs

    def backward(self, output_gradients):
        """
        Backward propagate through this layer. Gradients
        will be propagated by a transpose convolution with the kernels.
        :param output_gradients: The loss gradients respecting layer outputs
        :return: The partial derivatives of the loss with respect to inputs
        """

        # Calculate the kernels gradients
        self._kernels['grads'] = None

        # Calculate the biases gradients
        self._biases['grads'] = None