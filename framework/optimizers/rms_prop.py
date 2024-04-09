"""
Module for the root-mean-square optimizer class.
"""

# Avoid circular imports and retain type hinting
from __future__ import annotations

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.optimizers import Optimizer
from framework.optimizers.schedules import Schedule
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from framework.model import Model


class RMSProp(Optimizer):
    """
    Root-mean-square propagation optimization algorithm object to train a
    neural network. This algorithm works by updating the parameters
    proportional to their gradients and inversely to the square root of
    the gradient variances proxy.
    """

    # Small value to prevent division by zero
    _EPSILON = 1e-4

    def __init__(self, beta: float, schedule: Schedule) -> None:
        """
        Create a root-mean-square propagation optimizer for a model.
        :param beta: The weight of the previous mean square term
        :param schedule: The schedule to adjust the learning rate
        """

        # Call the superclass initializer
        super().__init__(schedule)

        # Save the defined optimization parameters
        self._BETA = beta

        # Define the model mean squares cache
        self._variances = {}

    def _calculate_variances(
            self,
            variances: ndarray,
            gradients: ndarray) -> ndarray:
        """
        Calculate the current gradient variances of a parameter.
        :param variances: The last calculated parameter variances
        :param gradients: The loss gradients respecting the parameters
        :return: The new variances of a parameter
        """

        # Calculate the new mean squares term
        beta = self._BETA
        if variances is not None:
            variances = beta * variances + (1 - beta) * (gradients ** 2)
        else:
            variances = gradients ** 2

        return variances

    def update(self, model: Model, epoch: int) -> Model:
        """
        Update the layer passed parameters using the algorithm.
        :param model: The model with parameters to update
        :param epoch: The epoch of the optimization step
        :return: The newly updated model
        """

        # Define the model variances cache
        model_id = model.MODEL_ID
        model_variances = self._variances.get(model_id, {})
        self._variances[model_id] = model_id

        # Update all parameters in all layers
        for layer in model.LAYERS:
            for parameter in layer.parameters.values():
                # Check that the key and gradients are defined
                parameter_id = parameter['id']
                gradients = parameter['grads'][:]  # None cannot be indexed

                # Recalculate the parameter variances
                variances = model_variances.get(parameter_id, None)
                variances = self._calculate_variances(variances, gradients)
                self._variances[model_id][parameter_id] = variances

                # Update the current parameter
                alpha = self._SCHEDULE.calculate_alpha(epoch)
                deltas = gradients / (np.sqrt(variances) + RMSProp._EPSILON)
                parameter['values'] -= alpha * deltas
                parameter['gradients'] = None

        return model
