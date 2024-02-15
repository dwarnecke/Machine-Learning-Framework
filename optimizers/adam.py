"""
Adam optimizer to train a model.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class Adam:
    def __init__(self, beta1: float, beta2: float, epsilon = 1e-4):
        """
        Create a root-mean-square propagation optimizer for a model.
        :param beta1: The weight of the previous mean gradients term
        :param beta2: The weight of the previous mean square gradients term
        :param epsilon: The division by zero offset term
        """

        # Save the defined optimization parameters
        self._BETA1 = beta1
        self._BETA2 = beta2
        self._EPSILON = epsilon

        # Define the term and iteration number caches
        self._means_cache = {}
        self._variances_cache = {}
        self._iteration_nums_cache = {}

    def calculate_adjustment(
            self,
            parameter_id: str,
            parameter_grads: np.ndarray,
            learning_rate: float) -> np.ndarray:
        """
        Calculate the adjustment term for optimizing any parameter.
        :param parameter_id: The parameter identification string for the cache
        :param parameter_grads: The loss gradients respecting parameters
        :param learning_rate: The rate at which to change the parameters by
        :return: The values to update the parameters by
        """

        # Retrieve the cached parameter terms
        prev_means = self._means_cache.get(parameter_id, None)
        prev_variances = self._variances_cache.get(parameter_id, None)
        curr_iter = self._iteration_nums_cache.get(parameter_id, 1)

        # Calculate the new mean and variance terms
        if not prev_means is None and not prev_variances is None:
            means_partial = self._BETA1 * prev_means
            variances_partial = self._BETA2 * prev_variances
        else:
            means_partial = np.zeros_like(parameter_grads)
            variances_partial = np.zeros_like(parameter_grads)
        means = means_partial + (1 - self._BETA1) * parameter_grads
        variances = \
            variances_partial \
            + (1 - self._BETA2) * (parameter_grads ** 2)

        # Cache the current terms for later use
        self._means_cache[parameter_id] = means
        self._variances_cache[parameter_id] = variances
        self._iteration_nums_cache[parameter_id] = curr_iter

        # Rescale the terms to account for bias
        corrected_means = means / (1 - self._BETA1 ** curr_iter)
        corrected_variances = variances / (1 - self._BETA2 ** curr_iter)

        # Calculate the parameter adjustment term
        parameter_adjustment = \
            learning_rate * corrected_means \
            / np.sqrt(corrected_variances + self._EPSILON)

        return parameter_adjustment  # Return the parameter adjustment
