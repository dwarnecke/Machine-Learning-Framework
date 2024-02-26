"""
Module for the momentum optimizer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '2.0'

import numpy as np


class Momentum:
    """
    Momentum optimization algorithm object to train a neural network.
    This algorithm works by updating the parameters proportional to their
    mean gradients.
    """

    def __init__(self, beta: float):
        """
        Create a momentum optimizer for a model.
        :param beta: The weight of the previous momentum term
        """

        # Save the defined beta
        self._BETA = beta

        # Define the model momentum cache
        self._momentum_cache = {}

    def _calculate_momentum(
            self,
            network_id: str,
            gradients: np.ndarray) -> np.ndarray:
        """
        Calculate the current gradient momentum of a parameter.
        :param network_id: The parameter network identification string
        :param gradients: The loss gradients respecting the parameters
        :return: The current momentum of the parameter
        """

        # Get the previous momentum term
        momentum = self._momentum_cache.get(network_id)

        # Calculate the new momentum term
        if momentum is not None:
            momentum = (
                self._BETA * momentum
                + (1 - self._BETA) * gradients)
        else:
            momentum = gradients

        return momentum

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

            # Calculate and save the current momentum
            momentum = self._calculate_momentum(network_id, gradients)
            self._momentum_cache[network_id] = momentum

            # Update the current parameter
            parameter['values'] -= alpha * momentum
            parameter['gradients'] = None

        return parameters
