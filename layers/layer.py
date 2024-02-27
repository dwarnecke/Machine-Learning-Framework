"""
Module for the parent layer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np

from optimizers.optimizer import Optimizer

class Layer:
    """
    Parent class for all layer classes. This layer should never be
    instantiated in an actual neural network.
    """

    def __init__(self, is_trainable: bool = True):
        """
        Create a dummy parent layer for a machine learning model.
        :param is_trainable: If the layer is trainable or not
        """

        # Define the layer architecture constants
        self.IS_TRAINABLE = is_trainable
        self.OUTPUT_SHAPE = None

        # Define the parameter dictionary
        if self.IS_TRAINABLE: self.parameters = {}

    def initialize(self, input_shape: tuple) -> None:
        pass

    def forward(self, layer_inputs: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        pass
