"""
Module for the adam optimizer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '2.0'

import numpy as np

from optimizers.optimizer import Optimizer

class Adam(Optimizer):
    """
    Adam optimization algorithm object to train a neural network. This
    algorithm works by updating the parameters proportional to the gradient
    means and inversely to the square root of the gradient variances proxy.
    """

    _EPSILON = 1e-4

    def __init__(self, beta1: float, beta2: float):
        """
        Create an adam optimizer for a model.
        :param beta1: The weight of the previous mean gradients term
        :param beta2: The weight of the previous mean square gradients term
        """

        # Save the defined optimization parameters
        self._BETA1 = beta1
        self._BETA2 = beta2

        # Define the term and iteration number caches
        self._means_cache = {}
        self._variances_cache = {}
        self._iteration_cache = {}

    def _calculate_means(
            self,
            network_id: str,
            gradients: np.ndarray) -> np.ndarray:
        """
        Calculate the current gradient means of a parameter.
        :param network_id: The parameter network identification string
        :param gradients: The loss gradients respecting the parameters
        :return: The current gradient means of the parameter
        """

        # Get the previous means term
        means = self._means_cache.get(network_id)

        # Calculate the new means term
        if means is not None:
            means = self._BETA1 * means + (1 - self._BETA1) * gradients
        else:
            means = (1 - self._BETA1) * gradients

        return means

    def _calculate_variances(
            self,
            network_id: str,
            gradients: np.ndarray) -> np.ndarray:
        """
        Calculate the current gradient variances of a parameter.
        :param network_id: The parameter network identification string
        :param gradients: The loss gradients respecting the parameters
        :return: The current mean square of a parameter
        """

        # Get the previous mean squares term
        variances = self._variances_cache.get(network_id, None)

        # Calculate the new mean squares term
        if variances is not None:
            variances = (
                    self._BETA2 * variances
                    + (1 - self._BETA2) * (gradients ** 2))
        else:
            variances = (1 - self._BETA2) * (gradients ** 2)

        return variances

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
                network_id = parameter['id']
                gradients = parameter['gradients']
            except KeyError:
                raise ValueError("Parameter gradients must be defined.")

            # Calculate and save the current algorithm terms
            means = self._calculate_means(network_id, gradients)
            variances = self._calculate_variances(network_id, gradients)
            self._means_cache[network_id] = means
            self._variances_cache[network_id] = variances

            # Correct the terms by their zero bias
            iteration = self._iteration_cache.get(network_id, 0) + 1
            corrected_means = means / (1 - self._BETA1 ** iteration)
            corrected_variances = variances / (1 - self._BETA2 ** iteration)
            self._iteration_cache[network_id] = iteration

            # Update the current parameter
            parameter['values'] -= (
                    alpha * corrected_means
                    / (np.sqrt(corrected_variances) + Adam._EPSILON))
            parameter['gradients'] = None

        return parameters