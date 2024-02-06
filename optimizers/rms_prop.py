"""
Root-mean-square propagation optimizer to train a model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class RMSProp:
    def __init__(self, beta: float, epsilon = 1e-8):
        """
        Create a root-mean-square propagation optimizer for a model.
        :param beta: The weight of the previous mean square term
        :param epsilon: The division by zero offset term
        """

        # Save the defined optimization parameters
        self._BETA = beta
        self._EPSILON = epsilon

        self._mean_squares_cache = {}  # Define the model mean squares cache

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
        prev_vars = self._mean_squares_cache.get(parameter_id, None)

        # Calculate the current gradient variances
        if prev_vars is not None:
            vars_partial = self._BETA * prev_vars
            grads_partial = (1 - self._BETA) * (parameter_grads ** 2)
            curr_vars = vars_partial + grads_partial
        else:
            curr_vars = parameter_grads ** 2

        # Cache the variances for later use
        self._mean_squares_cache[parameter_id] = curr_vars

        # Calculate the adjustment term
        parameter_adjustment = \
            learning_rate * parameter_grads \
            / np.sqrt(curr_vars + self._EPSILON)

        return parameter_adjustment  # Return the parameter adjustment
