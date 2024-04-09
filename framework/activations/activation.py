"""
Parent activation for all other activation functions.
"""

__author__ = 'Dylan Warnecke'

from numpy import ndarray


class Activation:
    """
    Abstract activation function to delineate transformed values. This class
    should never be actually instantiated.
    """

    def calculate(self, inputs: ndarray) -> ndarray:
        """
        Calculate the activation values of many inputs.
        :param inputs: The inputs to the activation function
        :return: The activation values of the inputs
        """
        pass

    def differentiate(self, inputs: ndarray) -> ndarray:
        """
        Differentiate the function at many inputs.
        :param inputs: The inputs to the activation function
        :return: The point derivatives of the inputs
        """
        pass

    def serialize(self) -> dict:
        """
        Serialize the activation function into a transmittable form.
        :return: The activation function parameters
        """
        pass