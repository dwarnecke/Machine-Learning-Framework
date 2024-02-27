"""
Neural network models to fit complex data using supervised learning.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np

from layers.input_layer import InputLayer
from layers.layer import Layer
from losses.loss import Loss
from optimizers.optimizer import Optimizer
from optimizers.gradient_descent import GradientDescent

class Model:
    """
    Neural network model to predict and classify data. Many layers can
    compose this model.
    """

    def __init__(self, loss: Loss, *model_layers: Layer) -> None:
        """
        Create the neural network model.
        :param loss: The loss function to optimize the model with
        :param optimizer: The optimizer to update the parameters with
        :param model_layers: The layers to be implemented into the model
        """

        # Check and define the model layers
        for layer_idx, layer in enumerate(model_layers):
            if layer_idx != 0 and isinstance(layer, InputLayer):
                raise ValueError("Only the first layer can be an input layer.")
            elif layer_idx == 0 and not isinstance(layer, InputLayer):
                raise ValueError("The first layer must be an input layer.")
        self._LAYERS = model_layers

        # Define the model loss function
        self._LOSS = loss

        # Initialize each of the layers
        input_shape = model_layers[0].OUTPUT_SHAPE
        for layer in model_layers[1:]:
            layer.initialize(input_shape)

            # Output of one layer is the input of the next
            if layer.OUTPUT_SHAPE is not None:
                input_shape = layer.OUTPUT_SHAPE

    def predict(self, examples: np.ndarray) -> np.ndarray:
        """
        Forward propagate through the model to get the respective outputs of
        the given inputs given the current model parameters.
        :param examples: The feature inputs to the model
        :return: The respective model predictions of the inputs
        """

        # Forward propagate through model
        inputs_propagated = examples
        for layer in self._LAYERS:
            inputs_propagated = layer.forward(
                inputs_propagated,
                in_training=False)

        return inputs_propagated

    def _improve(
            self,
            batch_examples: np.ndarray,
            batch_labels: np.ndarray,
            alpha: float,
            optimizer: Optimizer) -> None:
        """
        Complete an optimization step using a batch of examples and an
        optimization algorithm.
        :param batch_examples: The examples to use in the optimization step
        :param batch_labels: The truth labels of the batch examples
        :param alpha: The learning rate to change the parameters by
        :param optimizer: The optimizer algorithm to train the model
        """

        # Forward propagate through the model
        forward_outputs = batch_examples
        for layer in self._LAYERS:
            forward_outputs = layer.forward(forward_outputs, in_training=True)

        # Backward propagate through the model
        backward_gradients = self._LOSS.gradate(forward_outputs, batch_labels)
        for layer in reversed(self._LAYERS):
            backward_gradients = layer.backward(backward_gradients)

        # Update the layers with the parameter gradients
        for layer in self._LAYERS:
            if layer.IS_TRAINABLE:
                optimizer.update_parameters(layer.parameters, alpha)

    def fit(
            self,
            examples: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            epochs: int,
            alpha: float,
            optimizer: Optimizer = None) -> np.ndarray:
        """
        Optimize the model using batch gradient descent and the compilation
        parameters.
        :param examples: The training examples to fit the model on
        :param labels: The ground truth labels for the samples provided
        :param batch_size: The size of each batch in training
        :param epochs: The number of epochs to train the model for
        :param alpha: The learning rate to change the parameters by
        :param optimizer: The optimizer algorithm to train the model
        :return: The training history of the model
        """

        # Check that the batch size and epochs are positive
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        elif epochs < 1:
            raise ValueError("Epochs must be positive.")

        # Default to gradient descent when no optimizer is given
        if optimizer is None:
            optimizer = GradientDescent()

        # Define the training loss history
        training_history = np.empty(epochs)

        # Properly define the training batch size
        n_examples = examples.shape[0]
        if n_examples / 2 < batch_size:
            batch_size = n_examples

        # Message the model is building
        print(f"Fitting the model over {epochs} epochs:")

        # Fit the model using batch descent
        generator = np.random.default_rng()
        for epoch in range(epochs):
            # Shuffle the training examples and labels
            shuffling_indices = generator.permutation(n_examples)
            permuted_examples = examples[shuffling_indices]
            permuted_labels = labels[shuffling_indices]

            for batch_idx in range(n_examples // batch_size):
                # Index the batch from training examples
                lower_idx = batch_size * batch_idx
                upper_idx = batch_size * (batch_idx + 1)
                batch_examples = permuted_examples[lower_idx: upper_idx]
                batch_labels = permuted_labels[lower_idx: upper_idx]

                # Complete an optimization step using the batch
                self._improve(batch_examples, batch_labels, alpha, optimizer)

            # Calculate and cache the current model metrics
            current_outputs = self.predict(examples)
            current_loss = self._LOSS.calculate(current_outputs, labels)
            training_history[epoch] = current_loss

            # Print the current model performance
            print(f"Epoch {epoch + 1}: loss {round(current_loss, 4)}")

        # Message that the model is trained
        print(f"Model is fit after {epochs} epochs.\n")

        return training_history
