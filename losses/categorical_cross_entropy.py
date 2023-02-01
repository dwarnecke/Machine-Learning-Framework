"""
The categorical cross entropy loss function.

This loss function should be used when there is only one model output category
that each input case falls into.
"""

import numpy as np


def calculate_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Calculate the softmax normalization of given input logits.
    :param logits: The values to be normalized
    :return: The softmax normalization of the input logits
    """
    return np.exp(logits) / np.sum(np.exp(logits), axis=1)


class CategoricalCrossEntropy:
    def __init__(self, from_logits=False):
        """
        Define the loss being used to train a model
        :param from_logits: Whether the input to the loss is in logits or not
        """

        # Call the super initializer
        super().__init__()

        # Set if the loss calculation is from logit form or not
        self._FROM_LOGITS = from_logits

    def calculate(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the average categorical cross-entropy loss of a group of
        model sample predictions.
        :param outputs: The example model outputs to calculate loss of
        :param labels: The ground truth classifications for each sample
        :return: The average loss of all the input model predictions
        """

        # Check that the predictions and labels are the same shape
        if np.shape(outputs) != np.shape(labels):
            raise ValueError("Predictions and labels need the same shape.")

        # Check that the predictions and labels are only two-dimensional
        if np.ndim(outputs) != 2:
            raise ValueError("Predictions and labels must be two dimensional.")

        # Calculate the softmax activated values if necessary
        if self._FROM_LOGITS:
            predictions = calculate_softmax(outputs)
        else:
            predictions = outputs

        # Calculate the categorical cross entropy loss
        n_examples = np.shape(labels)[0]
        mean_loss = -np.sum(labels * np.log(predictions)) / n_examples

        return mean_loss  # Return the loss of the examples

    def gradate(self, outputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the categorical cross-entropy loss with
        respect to the model sample predictions.
        :param outputs: The sample model outputs to calculate loss of
        :param labels: The ground truth labels for each sample
        :return: The partial derivatives of the loss respecting model output
        """

        # Check that arguments are valid
        if type(outputs) != np.ndarray or type(labels) != np.ndarray:
            raise TypeError("Outputs and labels must be numpy arrays.")
        elif outputs.shape != labels.shape:
            raise ValueError("Outputs and labels need the same shape.")
        elif np.ndim(outputs) != 2:
            raise ValueError("Outputs and labels must be two dimensional.")

        # Calculate the loss gradient with respect to the model output
        n_examples = np.shape(labels)[0]
        if self._FROM_LOGITS:
            predictions = calculate_softmax(outputs)
            output_gradients = (predictions - labels) / n_examples
        else:
            output_gradients = -labels / (outputs * n_examples)

        return output_gradients  # Return the prediction gradients
