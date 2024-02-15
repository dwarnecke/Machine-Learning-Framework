"""
Parent class for all model layer classes.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'


class Layer:
    def __init__(self, is_trainable: bool = True):
        """
        Create a dummy parent layer for a machine learning model. This class
        should not be instantiated but instead used simply as the parent of
        all layers.
        :param is_trainable: If the layer is trainable or not
        """

        # Define the layer architecture parameters
        self.IS_TRAINABLE = is_trainable

        self._network_id = None  # Define the network identification key
        self._is_compiled = False  # Define the layer compilation flag
