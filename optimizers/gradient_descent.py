"""
Module for the dummy gradient descent optimizer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np

from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    """
    Gradient descent optimization algorithm object to train a neural
    network. This is the standard gradient descent algorithm.
    """

    def __init__(self):
        """
        Create a gradient descent optimizer for a model. There are no
        instance attributes of a gradient descent optimizer.
        """
        pass

    def update_parameters(self, parameters: dict, alpha: float) -> dict:
        """
        Update the layer passed parameters using the algorithm.
        :param parameters: The layer parameter dictionary to update
        :param alpha: The learning rate to change the parameters by
        :return: The newly updated layer parameter dictionary
        """

        # Check that the learning rate is positive
        if alpha <= 0:
            raise ValueError("Learning rate must be positive.")

        # Update every parameter in the layer
        for parameter in parameters.values():
            # Check that the id and gradients are defined
            try:
                gradients = parameter['gradients']
            except KeyError:
                raise ValueError("Parameter gradients must be defined.")

            # Update the current parameter
            parameter['values'] -= alpha * gradients
            parameter['gradients'] = None

        return parameters