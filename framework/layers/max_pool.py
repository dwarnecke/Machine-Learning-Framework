"""
Maximum pooling down-sampling layer in a neural network.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.layers import Layer


def _pool(inputs: ndarray, size: int) -> tuple:
    """
    Perform maximum pooling down-sampling over many windows of values.
    :param inputs: The values to be pooled
    :param size: The size of the pooling window
    :return: The pooling values and the respective input mask
    """

    # Define the spatial dimensions for convenience
    inputs_height = inputs.shape[1]
    inputs_width = inputs.shape[2]

    # Create empty pool and mask arrays
    pools_shape = list(inputs.shape)
    pools_shape[1] = 1 + (inputs_height - 1) // size
    pools_shape[2] = 1 + (inputs_width - 1) // size
    pools = np.empty(pools_shape)
    masks = np.empty_like(inputs)

    # Pool by maximum values and save the masks
    for y in range(0, inputs_width, size):
        for x in range(0, inputs_height, size):
            windows = inputs[..., y:(y + size), x:(x + size), :]

            # Pool the windows by maximum value
            maxes = np.max(windows, axis=(1, 2), keepdims=True)
            pools[..., [[int(y / size)]], [[int(x / size)]], :] = maxes

            # Compare the inputs to these maximums
            equalities = (windows == maxes)
            masks[..., y:(y + size), x:(x + size), :] = equalities

    return pools, masks


def _unpool(inputs: ndarray, masks: ndarray, size: int) -> ndarray:
    """
    Upsample pooled inputs at positions determined by the pooling masks.
    :param inputs: The values to be unpooled
    :param masks: The maximum value pooling masks
    :param size: The size of the pooling windows
    :return: The unpooled inputs
    """

    # Create empty dispersion arrays
    dispersions = np.empty_like(masks)

    # Unpool the inputs to their masked positions
    masks_height = masks.shape[1]
    masks_width = masks.shape[2]
    for y in range(0, masks_height, size):
        for x in range(0, masks_width, size):
            maxes = inputs[..., [[int(y / size)]], [[int(x / size)]], :]
            indices = masks[..., y:(y + size), x:(x + size), :]

            # Unpool the maximums to their pooling indices
            peeks = maxes * indices
            dispersions[..., y:(y + size), x:(x + size), :] = peeks

    return dispersions


class MaxPool(Layer):
    """
    Maximum pooling down-sampling layer for a neural network. These layers
    reduce the dimensionality of example data by several factors
    through propagating only the maximum values out of the spatial neighbors.
    """

    # Define the number of max pool layers created
    layers = 0

    def __init__(self, pool_shape: int) -> None:
        """
        Create a maximum pooling layer for a neural network.
        :param pool_shape: The height and width of the pooling window
        """

        # Call the superclass initializer
        super().__init__()

        # Define the network identification key
        MaxPool.layers += 1
        self.LAYER_ID = 'MaxPool' + str(MaxPool.layers)

        # Check that the pool size is positive
        if pool_shape < 1:
            raise ValueError("Pool size must be positive.")
        self._POOL_SHAPE = pool_shape

        # Define the layer pooling masks
        self._input_masks = None

    def forward(self, input_layers: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param input_layers: The inputs to this pooling layer
        :return: The outputs given the inputs
        """

        # Pool the inputs by maximum value
        output_layers, masks = _pool(input_layers, self._POOL_SHAPE)

        # Save the masks for backward propagation
        self._input_masks = masks

        return output_layers

    def backward(self, output_grads: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param output_grads: The loss gradients respecting layer outputs
        :return: The partial loss derivatives with respect to layer inputs
        """

        # Check that forward propagation has been completed
        masks = self._input_masks[:]  # None cannot be indexed
        self._input_masks = None  # Erase the masks to prevent repeat use

        # Unpool the gradients
        input_grads = _unpool(output_grads, masks, self._POOL_SHAPE)

        return input_grads

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """

        layer = {'type': 'max_pool', 'pool_shape': self._POOL_SHAPE}

        return layer
