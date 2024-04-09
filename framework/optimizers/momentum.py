"""
Module for the momentum optimizer class.
"""

# Avoid circular imports and retain type hinting
from __future__ import annotations

__author__ = 'Dylan Warnecke'

from numpy import ndarray
from framework.optimizers import Optimizer
from framework.optimizers.schedules import Schedule
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from framework.model import Model


class Momentum(Optimizer):
    """
    Momentum optimization algorithm object to train a neural network.
    This algorithm works by updating the parameters proportional to their
    mean gradients.
    """

    def __init__(self, beta: float, schedule: Schedule) -> None:
        """
        Create a momentum optimizer for a model.
        :param beta: The weight of the previous momentum term
        :param schedule: The schedule to adjust the learning rate
        """

        # Call the superclass initializer
        super().__init__(schedule)

        # Save the defined beta
        self._BETA = beta

        # Define the moments caches for the models
        self._moments = {}

    def _calculate_momentum(
            self,
            momentum: ndarray,
            gradients: ndarray) -> ndarray:
        """
        Calculate the new gradient momentum of a parameter.
        :param momentum: The last calculated parameter moments
        :param gradients: The loss gradients respecting the parameters
        :return: The new moments of the parameter
        """

        # Calculate the new momentum term
        beta = self._BETA
        if momentum is not None:
            momentum = beta * momentum + (1 - beta) * gradients
        else:
            momentum = gradients

        return momentum

    def update(self, model: Model, epoch: int) -> Model:
        """
        Update the layer passed parameters using the algorithm.
        :param model: The model with parameters to update
        :param epoch: The epoch of the optimization step
        :return: The newly updated model
        """

        # Define the model moments cache
        model_id = model.MODEL_ID
        model_moments = self._moments.get(model_id, {})
        self._moments[model_id] = model_moments

        # Update all parameters in all layers
        for layer in model.LAYERS:
            for parameter in layer.parameters.values():
                # Check that the key and gradients are defined
                parameter_id = parameter['id']
                gradients = parameter['grads'][:]  # None cannot be indexed

                # Recalculate the parameter momentum
                moments = model_moments.get(parameter_id, None)
                moments = self._calculate_momentum(moments, gradients)
                self._moments[model_id][parameter_id] = moments

                # Update the current parameter
                alpha = self._SCHEDULE.calculate_alpha(epoch)
                parameter['values'] -= alpha * moments
                parameter['gradients'] = None

        return model
