"""
Parent optimizer for all other optimizers.
"""

# Avoid circular imports and retain type hinting
from __future__ import annotations

__author__ = 'Dylan Warnecke'

from framework.optimizers.schedules.schedule import Schedule
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from framework.model import Model

class Optimizer:
    """
    Abstract class optimizer for training a machine learning model. This
    class should never be actually instantiated.
    """

    def __init__(self, schedule: Schedule) -> None:
        """
        Create the parent optimizer for a machine learning algorithm.
        :param schedule: The schedule to adjust the learning rate
        """

        # Define the common optimizer attributes
        self._SCHEDULE = schedule

    def update(self, model: Model, epoch: int) -> Model:
        """
        Update the model parameters using the optimization algorithm.
        :param model: The model with parameters to update
        :param epoch: The epoch of the optimization step
        :return: The newly updated model
        """
        pass