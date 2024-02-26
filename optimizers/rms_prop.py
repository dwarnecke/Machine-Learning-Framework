"""
Module for the root-mean-square optimizer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '2.0'

import numpy as np


class RMSProp:
    """
    Root-mean-square propagation optimization algorithm object to train a
    neural network. This algorithm works by updating the parameters
    proportional to their gradients and inversely to the square root of
    the gradient variances proxy.
    """

    # Small value to prevent division by zero
    _EPSILON = 1e-4

    def __init__(self, beta: float):
        """
        Create a root-mean-square propagation optimizer for a model.
        :param beta: The weight of the previous mean square term
        """

        # Save the defined optimization parameters
        self._BETA = beta

        # Define the model mean squares cache
        self._variances_cache = {}

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
                self._BETA * variances
                + (1 - self._BETA) * (gradients ** 2))
        else:
            variances = gradients ** 2

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

            # Calculate and save the current variances
            variances = self._calculate_variances(network_id, gradients)
            self._variances_cache[network_id] = variances

            # Update the current parameter
            parameter['values'] -= (
                    alpha * gradients
                    / (np.sqrt(variances) + RMSProp._EPSILON))
            parameter['gradients'] = None

        return parameters