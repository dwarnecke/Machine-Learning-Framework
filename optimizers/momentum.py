"""
Momentum optimizer to train a model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class Momentum:
    def __init__(self, beta: float):
        """
        Create a momentum optimizer for a model.
        :param beta: The weight of the previous momentum term
        """

        self._BETA = beta # Save the defined beta

        self._momentum_cache = {}  # Define the model momentum cache

    def calculate_delta(
            self,
            parameter_id: str,
            parameter_grads: np.ndarray,
            learning_rate: float) -> np.ndarray:
        """
        Calculate the adjustment term for optimizing any parameter.
        :param parameter_id: The identification string for the cache
        :param parameter_grads: The loss gradients respecting parameters
        :param learning_rate: The rate at which to change the parameters by
        :return: The values to update the parameters by
        """

        # Retrieve the previous parameter momentum
        prev_momentum = self._momentum_cache.get(parameter_id, None)

        # Calculate the momentum term of this iteration
        if prev_momentum is not None:
            momentum = (
                self._BETA * prev_momentum
                + (1 - self._BETA) * parameter_grads)
        else:
            momentum = parameter_grads

        # Cache the momentum for later use
        self._momentum_cache[parameter_id] = momentum

        # Rescale the momentum by the learning rate
        parameter_adjustment = learning_rate * momentum

        return parameter_adjustment  # Return the parameter adjustment
