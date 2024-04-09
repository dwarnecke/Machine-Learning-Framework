"""
Parent loss function for all other loss functions.
"""

__author__ = 'Dylan Warnecke'

from numpy import ndarray


class Loss:
    """
    Abstract loss function to train a machine learning model. This class
    should never be actually instantiated.
    """

    def calculate(self, outputs: ndarray, labels: ndarray) -> float:
        """
        Calculate the average loss of many model outputs.
        :param outputs: The model outputs to calculate loss of
        :param labels: The ground truth labels for the respective outputs
        :return: The average loss of the model outputs
        """
        pass

    def gradate(self, outputs: ndarray, labels: ndarray) -> ndarray:
        """
        Calculate the gradient of the loss with respect to the model outputs.
        :param outputs: The model outputs to calculate loss of
        :param labels: The ground truth labels for the respective outputs
        :return: The partial derivatives of the loss respecting the outputs
        """
        pass