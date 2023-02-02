"""
Initialization module for the layers package.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

from layers.dense import Dense
from layers.dropout import Dropout
from layers.input_layer import InputLayer
from layers.softmax import Softmax

# Ensure that these always match the names of the modules in this package
valid_layers = (Dense, Dropout, InputLayer, Softmax)


def verify_layer(layer: object) -> bool:
    """
    Verify if the object presented is a viable model layer.
    :param layer: The layer to be tested
    :return: If the object is one of the layers in the package
    """

    # Check if the layer passed is a valid layer
    if isinstance(layer, valid_layers):
        return True
    return False
