"""
Module for the abstract parent optimizer class.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

import numpy as np


class Optimizer:
    """
    Abstract parent class for all optimizers.
    """

    def update_parameters(self, parameters: dict, alpha: float) -> dict:
        pass