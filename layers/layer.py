"""
Module for the parent layer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'


class Layer:
    """
    Parent class for all layer classes. This layer should never be used in
    an instantiated neural network.
    """

    def __init__(self, is_trainable: bool = True):
        """
        Create a dummy parent layer for a machine learning model. This class
        should not be instantiated but instead used simply as the parent of
        all layers.
        :param is_trainable: If the layer is trainable or not
        """

        # Define the layer architecture parameters
        self.IS_TRAINABLE = is_trainable
        self.features = None

        # Define the layer identification key
        self._network_id = None