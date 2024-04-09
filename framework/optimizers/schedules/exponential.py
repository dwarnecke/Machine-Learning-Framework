"""
Exponential learning rate schedule for training a machine learning model.
"""

__author__ = 'Dylan Warnecke'

from framework.optimizers.schedules.schedule import Schedule


class Exponential(Schedule):
    """
    Exponential learning rate schedule for training a model. The learning
    rate is reduced by a constant factor every epoch.
    """

    def __init__(self, alpha: float, factor: float = 0.998) -> None:
        """
        Create an exponential learning rate schedule.
        :param alpha: The initial learning rate to adjust the parameters
        :param factor: The factor by which to reduce the learning rate by
        """

        # Call the superclass initializer
        super().__init__(alpha)

        # Check that the factor produces positive, decaying rates
        if not 0 < factor < 1:
            raise ValueError("The decay factor must be between zero and one.")

        # Define the exponential decay factor
        self._FACTOR = factor

    def calculate_alpha(self, epoch: int):
        """
        Calculate the learning rate to use in an epoch. This is the same as
        the initial learning rate.
        :param epoch: The epoch to be completed in training
        :return: The learning rate of this epoch
        """

        # Calculate the new learning rate
        alpha = self._INITIAL_ALPHA * self._FACTOR ** epoch

        return alpha

