"""
Module for the dummy gradient descent optimizer class.
"""

# Avoid circular imports and retain type hinting
from __future__ import annotations

__author__ = 'Dylan Warnecke'

from framework.optimizers import Optimizer
from framework.optimizers.schedules import Schedule
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from framework.model import Model

class GradientDescent(Optimizer):
    """
    Gradient descent optimization algorithm object to train a neural
    network. This is the standard gradient descent algorithm.
    """

    def __init__(self, schedule: Schedule):
        """
        Create a gradient descent optimizer for a model.
        :param schedule: The schedule to adjust the learning rate
        """

        # Call the superclass initializer
        super().__init__(schedule)

    def update(self, model: Model, epoch: int) -> Model:
        """
        Update the model parameter using the optimization algorithm.
        :param model: The model with parameters to update
        :param epoch: The epoch of the optimization step
        :return: The newly updated model
        """

        # Update all parameters in all layers
        for layer in model.LAYERS:
            for parameter in layer.parameters.values():
                # Check that the gradients are defined
                gradients = parameter['grads']  # None cannot be indexed

                # Update the current parameter
                alpha = self._SCHEDULE.calculate_alpha(epoch)
                parameter['values'] -= alpha * gradients
                parameter['gradients'] = None

        return model