"""
Parent schedule for all other learning rate schedules.
"""

__author__ = 'Dylan Warnecke'


class Schedule:
    """
    Abstract learning rate schedule for an optimizer. This class should
    never be actually instantiated.
    """

    def __init__(self, alpha: float):
        """
        Create the parent learning rate schedule for an optimizer.
        :param alpha: The initial learning rate to adjust the parameters
        """

        # Check that the learning rate is positive
        if alpha <= 0:
            raise ValueError("Learning rate must be positive.")

        # Define the initial learning rate
        self._INITIAL_ALPHA = alpha

    def calculate_alpha(self, epoch: int):
        """
        Calculate the learning rate to use in an epoch.
        :param epoch: The epoch to be completed in training
        :return: The learning rate of this epoch
        """
        pass