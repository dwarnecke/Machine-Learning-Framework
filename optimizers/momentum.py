"""
Momentum optimizer for a network layer parameter.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class Momentum:
    def __init__(self, beta: float):
        """
        Create a momentum optimizer for a network parameter.
        :param beta: The weight of the previous momentum term
        """

        self._BETA = beta # Save the defined beta

        self._momentum_cache = {}  # Define the model momentum cache

    def calculate_adjustment(
            self,
            parameter_id: str,
            parameter_gradients: np.ndarray,
            learning_rate: float) -> np.ndarray:
        """
        Calculate the adjustment term for optimizing the parameter.
        :param learning_rate: The rate at which to change the parameters by
        :param parameter_gradients: The loss gradients respecting parameters
        :param parameter_id: The identification string for the cache
        :return: The values to update the parameters by
        """

        # Retrieve the previous parameter momentum
        previous_momentum = self._momentum_cache.get(parameter_id, None)

        # Calculate the momentum term of this iteration
        if previous_momentum is not None:
            momentum_partial = self._BETA * previous_momentum
            gradient_partial = (1 - self._BETA) * parameter_gradients
            momentum = momentum_partial + gradient_partial
        else:
            momentum = parameter_gradients

        # Cache the momentum for later use
        self._momentum_cache[parameter_id] = momentum

        # Rescale the momentum by the learning rate
        parameter_adjustment = learning_rate * momentum

        return parameter_adjustment  # Return the parameter adjustment
