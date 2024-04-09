"""
Neural network models to fit complex data using supervised learning.
"""

__author__ = 'Dylan Warnecke'

import json
import numpy as np
from numpy import ndarray
from framework.layers.layer import Layer
from framework.layers.input import Input
from framework.losses.loss import Loss
from framework.optimizers.optimizer import Optimizer

generator = np.random.default_rng()


def shuffle_data(examples: ndarray, labels: ndarray) -> tuple:
    """
    Shuffle the examples and their labels in the same order.
    :param examples: The examples to shuffle
    :param labels: The labels of the examples to shuffle
    :return: The examples and labels shuffled on the first axis
    """

    # Check that the first axis dimensions of the arguments match
    if examples.shape[0] != labels.shape[0]:
        raise IndexError("Examples and labels first dimension must match.")
    quantity = examples.shape[0]

    # Shuffle the examples and their labels
    shuffling_indices = generator.permutation(quantity)
    shuffled_examples = examples[shuffling_indices]
    shuffled_labels = labels[shuffling_indices]

    return shuffled_examples, shuffled_labels


def batch_data(examples: ndarray, labels: ndarray, size: int) -> tuple:
    """
    Batch the examples and their respective labels into many groups.
    :param examples: The examples to batch
    :param labels: The labels of the examples to batch
    :param size: The size of each batch
    :return: The list of data batches
    """

    # Check that the batch size is positive
    if size < 1:
        raise ValueError("Batch size must be positive.")
    
    # Check that the first axis dimensions of the arguments match
    if examples.shape[0] != labels.shape[0]:
        raise IndexError("Examples and labels first dimension must match.")
    quantity = examples.shape[0]

    # Adjust the batching size if useful
    if quantity / 2 < size:
        size = quantity

    # Batch the data together by the indices
    example_batches = [examples[i:(i+size)] for i in range(0, quantity, size)]
    label_batches = [labels[i:(i+size)] for i in range(0, quantity, size)]

    # Zip the examples and their respective labels together
    batches = tuple(zip(example_batches, label_batches))

    return batches


class Model:
    """
    Neural network model to predict and classify data. Many layers can
    compose this model.
    """

    # Define the number of models created
    models = 0

    def __init__(self, loss: Loss, *layers: Layer) -> None:
        """
        Create the neural network model.
        :param loss: The loss function to optimize the model with
        :param layers: The layers to be implemented into the model
        """

        # Define the model identification key
        Model.models += 1
        model_id = 'Model' + str(Model.models)
        self.MODEL_ID = model_id

        # Check that the only input layer is the first layer
        for layer_idx, layer in enumerate(layers):
            if layer_idx != 0 and isinstance(layer, Input):
                raise ValueError("Only the first layer can be an input layer.")
            elif layer_idx == 0 and not isinstance(layer, Input):
                raise ValueError("The first layer must be an input layer.")
        self.LAYERS = layers

        # Define the loss function
        self._LOSS = loss

        # Message that the model is created
        print(f"Model built with {len(self.LAYERS)} layers.\n")

    def _optimize(
            self,
            batch_examples: ndarray,
            batch_labels: ndarray,
            optimizer: Optimizer,
            epoch: int) -> None:
        """
        Complete an optimization step using a batch of examples and an
        optimization algorithm.
        :param batch_examples: The examples to use in the optimization step
        :param batch_labels: The truth labels of the batch examples
        :param optimizer: The optimizer algorithm to train the model
        :param epoch: The epoch number of the optimization step
        """

        # Forward propagate through the model
        forward_outputs = batch_examples
        for layer in self.LAYERS:
            forward_outputs = layer.forward(forward_outputs, in_training=True)

        # Backward propagate through the model
        backward_gradients = self._LOSS.gradate(forward_outputs, batch_labels)
        for layer in reversed(self.LAYERS):
            backward_gradients = layer.backward(backward_gradients)

        # Update the layers with the parameter gradients
        optimizer.update(self, epoch)

    def fit(
            self,
            examples: ndarray,
            labels: ndarray,
            batch_size: int,
            epochs: int,
            optimizer: Optimizer) -> list:
        """
        Optimize the model using batch gradient descent and the compilation
        parameters.
        :param examples: The training examples to fit the model on
        :param labels: The ground truth labels for the samples provided
        :param batch_size: The size of each batch in training
        :param epochs: The number of epochs to train the model for
        :param optimizer: The optimizer algorithm to train the model
        :return: The training history of the model
        """

        # Check that the batch size and epochs are positive
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        elif epochs < 1:
            raise ValueError("Epochs must be positive.")

        # Message that the model is being trained
        print(f"Fitting the model over {epochs} epochs:")

        # Fit the model using batch descent
        training_history = []  # Define the training history
        for epoch in range(epochs):
            # Randomly batch the examples together
            random_examples, random_labels = shuffle_data(examples, labels)
            batches = batch_data(random_examples, random_labels, batch_size)

            # Optimize the model using the batch
            for batch_examples, batch_labels in batches:
                self._optimize(batch_examples, batch_labels, optimizer, epoch)

            # Calculate and cache the current model metrics
            outputs = self.predict(examples)
            loss = self._LOSS.calculate(outputs, labels)
            training_history += [loss]

            # Message the current model performance
            print(f"Epoch {epoch + 1}: loss {round(loss, 4)}")

        # Message that the model is trained
        print(f"Model is fit after {epochs} epochs.\n")

        return training_history

    def predict(self, examples: ndarray) -> ndarray:
        """
        Forward propagate through the model to get the respective outputs of
        the given inputs given the current model parameters.
        :param examples: The feature inputs to the model
        :return: The respective model predictions of the inputs
        """

        # Forward propagate through model
        forward_outputs = examples
        for layer in self.LAYERS:
            forward_outputs = layer.forward(forward_outputs, in_training=False)

        return forward_outputs

    def save(self, name: str) -> dict:
        """
        Save the model architecture and parameters to a JSON file for later
        importation.
        :param name: The name for the new file that ends with '.json'
        :return: The dictionary of the model information
        """

        # Serialize all the model attributes
        model = {}
        for idx, layer in enumerate(self.LAYERS):
            model[idx] = layer.serialize()

        # Save the attributes to a file
        with open(name, 'w') as file:
            json.dump(model, file, indent=4)

        # Message that the model is saved
        print(f"Saved the model to {name}...")

        return model
    
