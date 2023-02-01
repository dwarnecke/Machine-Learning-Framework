"""
Parent class for all machine learning model layers
"""

__author__ = "Dylan Warnecke"

__version__ = '1.0'


class Layer:
    def __init__(self, is_trainable=True):
        """
        Create a dummy parent layer for a machine learning model. This class
        should not be instantiated but instead used simply as the parent of
        all layers.
        :param is_trainable: If the layer is trainable or not
        """

        # Define the layer compilation flag
        self._is_compiled = False

        # Set if the layer is trainable or compilable
        self.IS_TRAINABLE = is_trainable
