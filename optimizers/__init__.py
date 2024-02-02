"""
Initialization module for the optimizers package.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

from optimizers.momentum import Momentum

# Ensure that these always match the names of the modules in this package
valid_optimizers = (None, Momentum)