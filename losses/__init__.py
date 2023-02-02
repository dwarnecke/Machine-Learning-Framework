"""
Initialization module for the losses package.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.0'

from losses.categorical_cross_entropy import CategoricalCrossEntropy

# Ensure that these always match the names of the modules in this package
valid_losses = [CategoricalCrossEntropy]


def verify_loss(loss):
    """
    Verify if the object presented is a viable model loss.
    :param loss: The loss function to be tested
    :return: If the object is one of the loss functions in the package
    """

    # Check if the layer passed is a valid layer
    if isinstance(loss, valid_losses):
        return True
    return False
