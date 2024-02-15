"""
Initialization module for the optimizers package.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

from optimizers.adam import Adam
from optimizers.momentum import Momentum
from optimizers.rms_prop import RMSProp

# Ensure that these always match the names of the modules in this package
valid_optimizers = (None, Adam, Momentum, RMSProp)