"""
Initialization module for the activations package.
"""

__author__ = 'Dylan Warnecke'

from framework.activations.activation import Activation
from framework.activations.linear import Linear
from framework.activations.relu import Relu
from framework.activations.sigmoid import Sigmoid
from framework.activations.tanh import Tanh

# Activations grouped by how parameters are be initialized
xavier_activations = (Sigmoid, Tanh)
he_activations = (Relu,)
