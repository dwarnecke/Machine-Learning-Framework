"""
Module for the adam optimizer class.
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


class Adam(Optimizer):
    """
    Adam optimization algorithm object to train a neural network. This
    algorithm works by updating the parameters proportional to the gradient
    means and inversely to the square root of the gradient variances proxy.
    """

    _EPSILON = 1e-4

    def __init__(self, beta1: float, beta2: float, schedule: Schedule) -> None:
        """
        Create an adam optimizer for a model.
        :param beta1: The weight of the previous mean gradients term
        :param beta2: The weight of the previous mean square gradients term
        :param schedule: The schedule to adjust the learning rate
        """

        # Call the superclass initializer
        super().__init__(schedule)

        # Save the defined optimization parameters
        self._BETA1 = beta1
        self._BETA2 = beta2

        # Define the means and variances caches
        self._means = {}
        self._variances = {}

        # Define the optimization step number
        self._steps = {}

    def _calculate_means(self, means: ndarray, gradients: ndarray) -> ndarray:
        """
        Calculate the current gradient means of a parameter.
        :param means: The last calculated parameter moments
        :param gradients: The loss gradients respecting the parameters
        :return: The current gradient means of the parameter
        """

        # Calculate the new means term
        beta1 = self._BETA1
        if means is not None:
            means = beta1 * means + (1 - beta1) * gradients
        else:
            means = (1 - beta1) * gradients

        return means

    def _calculate_variances(
            self,
            variances: ndarray,
            gradients: ndarray) -> ndarray:
        """
        Calculate the current gradient variances of a parameter.
        :param variances: The last calculated parameter variances
        :param gradients: The loss gradients respecting the parameters
        :return: The current mean square of a parameter
        """

        # Calculate the new mean squares term
        beta2 = self._BETA2
        if variances is not None:
            variances = beta2 * variances + (1 - beta2) * (gradients ** 2)
        else:
            variances = (1 - beta2) * (gradients ** 2)

        return variances

    def update(self, model: Model, epoch: int) -> Model:
        """
        Update the layer passed parameters using the algorithm.
        :param model: The model with parameters to update
        :param epoch: The epoch of the optimization step
        :return: The newly updated model
        """

        # Define the model means and variances caches
        model_id = model.MODEL_ID
        model_means = self._means.get(model_id, {})
        model_variances = self._variances.get(model_id, {})
        self._means[model_id] = model_means
        self._variances[model_id] = model_variances

        # Define the number of optimization steps completed
        steps = self._steps.get(model_id, 0) + 1
        self._steps[model_id] = steps

        # Update all parameters in all layers
        for layer in model.LAYERS:
            for parameter in layer.parameters.values():
                # Check that the key and gradients are defined
                parameter_id = parameter['id']
                gradients = parameter['grads'][:]  # None cannot be indexed

                # Recalculate and correct the parameter means
                means = model_means.get(parameter_id, None)
                means = self._calculate_means(means, gradients)
                self._means[parameter_id] = means
                means *= 1 / (1 - self._BETA1 ** steps)

                # Recalculate and correct the parameter variances
                variances = model_variances.get(model_id, None)
                variances = self._calculate_variances(variances, gradients)
                self._variances[parameter_id] = variances
                variances *= 1 / (1 - self._BETA2 ** steps)

                # Update the current parameter
                alpha = self._SCHEDULE.calculate_alpha(epoch)
                deltas = means / (np.sqrt(variances) + Adam._EPSILON)
                parameter['values'] -= alpha * deltas
                parameter['gradients'] = None

        return model

