"""
Parent layer for all other layers.
"""

__author__ = 'Dylan Warnecke'

from numpy import ndarray


class Layer:
    """
    Abstract layer for a machine learning model. This class should never be
    actually instantiated.
    """

    def __init__(self):
        """
        Create the parent layer for a machine learning model.
        """

        # Define the layer identification key
        self.LAYER_ID = None

        # Define the layer parameters dictionary
        self.parameters = {}

    def forward(self, layers_in: ndarray, **kwargs) -> ndarray:
        """
        Forward propagate through this layer.
        :param layers_in: The inputs to this model layer
        :return: The calculated layer outputs given the current parameters
        """
        pass

    def backward(self, gradients_out: ndarray) -> ndarray:
        """
        Backward propagate through this layer.
        :param gradients_out: The loss gradients respecting the layer outputs
        :return: The partial loss derivatives with respect to layer inputs
        """
        pass

    def serialize(self) -> dict:
        """
        Serialize the layer into a transmittable form.
        :return: The layer attributes and parameters
        """
        pass

