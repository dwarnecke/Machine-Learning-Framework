"""
Root-mean-square propagation optimizer to train a model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class RMSProp:
    def __init__(self, beta: float, epsilon = 1e-4):
        """
        Create a root-mean-square propagation optimizer for a model.
        :param beta: The weight of the previous mean square term
        :param epsilon: The division by zero offset term
        """

        # Save the defined optimization parameters
        self._BETA = beta
        self._EPSILON = epsilon

        self._variances_cache = {}  # Define the model mean squares cache

    def calculate_adjustment(
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

        # Retrieve the previous parameter mean squares
        prev_variances = self._variances_cache.get(parameter_id, None)

        # Calculate the current gradient variances
        if prev_variances is not None:
            variances_partial = self._BETA * prev_variances
            grads_partial = (1 - self._BETA) * (parameter_grads ** 2)
            variances = variances_partial + grads_partial
        else:
            variances = parameter_grads ** 2

        # Cache the variances for later use
        self._variances_cache[parameter_id] = variances

        # Calculate the update term
        parameter_adjustment = \
            learning_rate * parameter_grads \
            / (np.sqrt(variances) + self._EPSILON)

        return parameter_adjustment  # Return the parameter adjustment
