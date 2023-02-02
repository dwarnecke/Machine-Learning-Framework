"""
Parent class for all model layer classes.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'


class Layer:
    def __init__(self, is_trainable=True):
        """
        Create a dummy parent layer for a machine learning model. This class
        should not be instantiated but instead used simply as the parent of
        all layers.
        :param is_trainable: If the layer is trainable or not
        """

        self._is_compiled = False  # Define the layer compilation flag
        self.IS_TRAINABLE = is_trainable  # Set if the layer is trainable
