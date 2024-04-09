"""
Initialization module for the layers package.
"""

__author__ = 'Dylan Warnecke'


# Avoid circular importation placing method above imports
def serialize_parameters(parameters: dict) -> dict:
    """
    Convert any layer parameters dictionary into JSON serializable form.
    :param parameters: The parameters dictionary to be serialized
    :return: The equivalent JSON serializable parameters dictionary
    """

    # Ensure that the serialization does not change the original copy
    parameters = parameters.copy()

    # Serialize every parameter passed
    for parameter in parameters.values():
        parameter['values'] = parameter['values'].tolist()
        del parameter['grads']

    return parameters


from framework.layers.layer import Layer
from framework.layers.convolution import Convolution
from framework.layers.dense import Dense
from framework.layers.dropout import Dropout
from framework.layers.flatten import Flatten
from framework.layers.input import Input
from framework.layers.max_pool import MaxPool
from framework.layers.softmax import Softmax
