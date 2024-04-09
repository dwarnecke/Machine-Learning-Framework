"""
Two-dimensional convolution layer in a machine learning model.
"""

__author__ = 'Dylan Warnecke'

import math
import numpy as np
from numpy import ndarray
from framework.activations.activation import Activation
from framework.activations.linear import Linear
from framework.layers import serialize_parameters
from framework.layers.layer import Layer
from framework.activations import relu


def _pad(inputs: ndarray, size: int) -> ndarray:
    """
    Pad zeros around images to prevent loss in convolution.
    :param inputs: The input data to be padded
    :param size: The number of zeros to pad the image with
    :return: The padded data of the images
    """

    # Define the padded inputs as a default array of zeroes
    padded_shape = list(inputs.shape)
    padded_shape[-3] += 2 * size
    padded_shape[-2] += 2 * size
    padded_inputs = np.zeros(padded_shape)

    # Replace the intermediate zeroes with the original data
    inputs_h = inputs.shape[-3]
    inputs_w = inputs.shape[-2]
    x_a, x_b = (size, size + inputs_w)
    y_a, y_b = (size, size + inputs_h)
    padded_inputs[..., y_a:y_b, x_a:x_b, :] = inputs

    return padded_inputs


def _convolve(inputs: ndarray, kernels: ndarray) -> ndarray:
    """
    Convolve many input channels and kernels. Kernels are assumed to have
    the same height and width.
    :param inputs: The inputs to use in the convolutions
    :param kernels: The kernels to use in the convolutions
    :return: The convolution feature maps of the inputs and kernels
    """

    # Define different dimensions for convenience
    inputs_h = inputs.shape[1]
    inputs_w = inputs.shape[2]
    channels_out = kernels.shape[0]
    kernel_size = kernels.shape[1]

    # Create empty feature map arrays
    maps_shape = list(inputs.shape)
    maps_shape[1] = inputs_h - (kernel_size - 1)
    maps_shape[2] = inputs_w - (kernel_size - 1)
    maps_shape[3] = channels_out
    feature_maps = np.empty(maps_shape)

    # Convolve the inputs with the kernels
    dot_axes = 2 * ((3, 2, 1),)
    for y in range(inputs_h - (kernel_size - 1)):
        for x in range(inputs_w - (kernel_size - 1)):
            windows = inputs[..., y:(y + kernel_size), x:(x + kernel_size), :]

            # Convolve the windows and the kernels into the maps
            convolutions = np.tensordot(windows, kernels, dot_axes)
            feature_maps[..., y, x, :] = convolutions

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
            kernel_shape: int,
            padding: bool,
            activation: Activation = relu):
        """
        Create a two-dimensional convolution layer for a neural network.
        :param channels: The number of output filters in the convolution
        :param kernel_shape: The height and width of the convolution kernels
        :param padding: If zero padding is added before convolution
        :param activation: The activation function of the layer
        """

        # Call the superclass initializer
        super().__init__()

        # Define the layer identification key
        Convolution.layers += 1
        self.LAYER_ID = 'Dense' + str(Convolution.layers)

        # Check that the number of channels is positive
        if channels < 1:
            raise ValueError("Number of channels must be positive.")
        else:
            self._CHANNELS = channels

        # Check that the kernel size is positive and odd
        if kernel_shape < 1:
            raise ValueError("Kernel shape must be positive.")
        elif kernel_shape % 2 != 1:
            raise ValueError("Kernel shape must be odd.")

        # Define the default activation as the linear function
        if activation is None:
            self._ACTIVATION = Linear()
        else:
            self._ACTIVATION = activation

        # Define the other layer hyperparameters
        self._KERNEL_SHAPE = kernel_shape
        self._PADDING = padding

        # Define the parameter dictionaries and identification keys
        self.parameters = {'kernels': {}, 'biases': {}}
        self.parameters['kernels']['id'] = self.LAYER_ID + '_kernels'
        self.parameters['biases']['id'] = self.LAYER_ID + '_biases'

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
        channels_in = inputs_shape[-1]
        channels_out = self._CHANNELS
        kernel_shape = self._KERNEL_SHAPE

        # Initialize the kernels using Xavier initialization
        generator = np.random.default_rng()
        scale = math.sqrt(2 / (channels_in * kernel_shape ** 2 + channels_out))
        size = (channels_out,) + 2 * (kernel_shape,) + (channels_in,)
        kernels = generator.normal(0, scale, size)
        self.parameters['kernels']['values'] = kernels

        # Initialize the biases at zero
        size = len(inputs_shape) * (1,) + (channels_out,)
        biases = np.zeros(size)
        self.parameters['biases']['values'] = biases

        return True

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to this convolution layer
        :return: The outputs given the inputs and current parameters
        """

        # Check that the layer is configured
        if not self._is_configured:
            inputs_shape = input_layers.shape[1:]
            self._is_configured = self._configure(inputs_shape)

        # Pad the inputs adequately
        padded_inputs = input_layers
        if self._PADDING:
            paring_size = int((self._KERNEL_SHAPE - 1) / 2)
            padded_inputs = _pad(input_layers, paring_size)

        # Convolve and delineate the inputs
        kernels = self.parameters['kernels']['values']
        biases = self.parameters['biases']['values']
        linear_mediums = _convolve(padded_inputs, kernels) + biases
        output_layers = self._ACTIVATION.calculate(linear_mediums)

        # Save the calculations for backpropagation later
        self._input_layers = input_layers
        self._linear_mediums = linear_mediums

        return output_layers

    def backward(self, output_grads) -> ndarray:
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
        linear_grads = output_grads * activation_grads

        # Pad the input values adequately for calculating kernel gradients
        if self._PADDING:
            padding_size = int((self._KERNEL_SHAPE - 1) / 2)
            padded_inputs = _pad(input_layers, padding_size)
        else:
            padded_inputs = input_layers

        # Pad the linear gradients adequately for calculating input gradients
        if self._PADDING:
            padding_size = int((self._KERNEL_SHAPE - 1) / 2)
            padded_grads = _pad(linear_grads, padding_size)
        else:
            padding_size = self._KERNEL_SHAPE - 1
            padded_grads = _pad(linear_grads, padding_size)

        # Calculate the kernels gradients
        outputs_transposed = linear_grads.swapaxes(0, -1)
        inputs_transposed = padded_inputs.swapaxes(0, -1)
        grads_transposed = _convolve(inputs_transposed, outputs_transposed)
        kernels_grads = grads_transposed.swapaxes(0, -1)
        self.parameters['kernels']['grads'] = kernels_grads

        # Calculate the biases gradients
        biases_grads = linear_grads.sum((0, 1, 2), keepdims=True)
        self.parameters['biases']['grads'] = biases_grads

        # Calculate the input gradients
        kernels = self.parameters['kernels']['values']
        kernels_flipped = kernels[:, ::-1, ::-1, :].swapaxes(0, -1)
        input_grads = _convolve(padded_grads, kernels_flipped)

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
            'type': 'convolution',
            'channels': self._CHANNELS,
            'kernel_shape': self._KERNEL_SHAPE,
            'padding': self._PADDING,
            'activation': activation,
            'parameters': parameters
        }

        return layer