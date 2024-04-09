"""
Module for the categorical cross entropy loss function.
"""

__author__ = 'Dylan Warnecke'

import numpy as np
from numpy import ndarray
from framework.losses.loss import Loss


def calculate_softmax(logits: ndarray) -> ndarray:
    """
    Calculate the softmax normalization of given input logits.
    :param logits: The values to be normalized
    :return: The softmax normalization of the input logits
    """

    epsilon = 1e-4  # Small value to offset division by zero
    clip = 1e2  # Large value to prevent exponential overflow

    # Calculate the softmax values of the logit values
    odds = np.exp(np.clip(logits, -clip, clip))
    probabilities = odds / (np.sum(odds, axis=1)[:, np.newaxis] + epsilon)

    return probabilities


class CategoricalCrossEntropy(Loss):
    """
    Categorical cross entropy loss. This loss function should be used when
    there is only one model output label that each input can be classified
    with.
    """

    def __init__(self, from_logits=False):
        """
        Define the loss being used to train a model
        :param from_logits: Whether the input to the loss is in logits or not
        """

        # Set if the loss calculation is from logit form or not
        self._FROM_LOGITS = from_logits

    def calculate(self, outputs: ndarray, labels: ndarray) -> float:
        """
        Calculate the average categorical cross-entropy loss of a group of
        model sample predictions.
        :param outputs: The example model outputs to calculate loss of
        :param labels: The ground truth classifications for each sample
        :return: The average loss of all the input model predictions
        """

        # Calculate the softmax activated values if necessary
        if self._FROM_LOGITS:
            predictions = calculate_softmax(outputs)
        else:
            predictions = outputs

        # Calculate the categorical cross entropy loss
        n_examples = np.shape(labels)[0]
        mean_loss = -np.sum(labels * np.log(predictions)) / n_examples

        return mean_loss

    def gradate(self, outputs: ndarray, labels: ndarray) -> ndarray:
        """
        Calculate the gradient of the categorical cross-entropy loss with
        respect to the model sample predictions.
        :param outputs: The sample model outputs to calculate loss of
        :param labels: The ground truth labels for each sample
        :return: The partial derivatives of the loss respecting model output
        """

        # Calculate the loss gradient with respect to the model output
        quantity = np.shape(labels)[0]
        if self._FROM_LOGITS:
            predictions = calculate_softmax(outputs)
            output_gradients = (predictions - labels) / quantity
        else:
            output_gradients = -labels / (outputs * quantity)

        return output_gradients
