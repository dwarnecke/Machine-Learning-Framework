"""
Constant learning rate schedule for training a machine learning model.
"""

__author__ = 'Dylan Warnecke'

from framework.optimizers.schedules.schedule import Schedule


class Constant(Schedule):
    """
    Constant learning rate schedule for training a model. The same, initial
    learning rate is used in every epoch.
    """

    def __init__(self, alpha: float):
        """
        Create a constant learning rate schedule.
        :param alpha: The initial learning rate to adjust the parameters
        """

        # Call the superclass initializer
        super().__init__(alpha)

    def calculate_alpha(self, epoch: int):
        """
        Calculate the learning rate to use in an epoch. This is the same as
        the initial learning rate.
        :param epoch: The epoch to be completed in training
        :return: The learning rate of this epoch
        """

        return self._INITIAL_ALPHA