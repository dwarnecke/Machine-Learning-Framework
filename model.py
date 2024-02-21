"""
Neural network models to fit complex data using supervised learning.
"""

__author__ = 'Dylan Warnecke'

__version__ = '1.1'

import numpy as np

import layers
import losses
import optimizers
from layers.input_layer import InputLayer


class Model:
    """
    Neural network model to predict and classify data. Many layers can
    compose this model.
    """

    def __init__(
            self,
            loss: losses.valid_losses,
            *model_layers: layers.valid_layers):
        """
        Create the neural network model.
        :param loss: The loss function to optimize the model with
        :param optimizer: The optimizer to update the parameters with
        :param model_layers: The layers to be implemented into the model
        """

        print("Model is being built...")  # Message the model is being built

        self._LOSS = loss  # Define the model loss function

        # Check the layer types are valid
        for layer_idx, layer in enumerate(model_layers):
            if layer_idx != 0 and isinstance(layer, InputLayer):
                raise ValueError("Only the first layer can be an input layer.")
            elif layer_idx == 0 and not isinstance(layer, InputLayer):
                raise ValueError("The first layer must be an input layer.")
        self._LAYERS = model_layers

        # Initialize each of the layers
        for layer_idx, layer in enumerate(model_layers):
            if layer_idx == 0:
                self._INPUT_UNITS = layer.features
            else:
                previous_features = model_layers[layer_idx - 1].features
                layer.initialize(previous_features, layer_idx)

        # Message that the model is built
        print(f"Model built with {len(self._LAYERS)} layers\n")

    def predict(self, examples: np.ndarray) -> np.ndarray:
        """
        Forward propagate through the model to get the respective outputs of
        the given inputs given the current model parameters.
        :param examples: The feature inputs to the model
        :return: The respective model predictions of the inputs
        """

        # Check that the input shape is consistent
        if examples.shape[-1] != self._INPUT_UNITS:
            raise ValueError("Number of inputs must be consistent.")

        # Forward propagate through model
        inputs_propagated = examples
        for layer in self._LAYERS:
            inputs_propagated = layer.forward(
                inputs_propagated,
                in_training=False)

        return inputs_propagated  # Return the model outputs

    def fit(
            self,
            examples: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            optimizer: optimizers.valid_optimizers = None) -> np.ndarray:
        """
        Optimize the model using batch gradient descent and the compilation
        parameters.
        :param examples: The training examples to fit the model on
        :param labels: The ground truth labels for the samples provided
        :param batch_size: The size of each batch in training
        :param epochs: The number of epochs to train the model for
        :param learning_rate: The rate at which to update the parameters at
        :return: The training history of the model
        :param optimizer: The optimizer algorithm to train the model
        """

        # Check that the batch size and epochs are positive
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        elif epochs < 1:
            raise ValueError("Epochs must be positive.")

        # Define the training loss history
        training_history = np.empty(epochs)

        # Properly define the training batch size
        n_examples = examples.shape[0]
        if n_examples / 2 < batch_size:
            batch_size = n_examples

        # Message the model is building
        print(f"Fitting the model over {epochs} epochs...")

        # Fit the model using batch descent
        generator = np.random.default_rng()
        for epoch in range(epochs):
            # Shuffle the training examples and labels
            shuffling_indices = generator.permutation(n_examples)
            permuted_examples = examples[shuffling_indices]
            permuted_labels = labels[shuffling_indices]

            for batch in range(n_examples // batch_size):
                # Index the batch from training examples
                lower_idx = batch_size * batch
                upper_idx = batch_size * (batch + 1)
                batch_examples = permuted_examples[lower_idx: upper_idx]
                batch_labels = permuted_labels[lower_idx: upper_idx]

                # Forward propagate through the model
                forward_inputs = batch_examples
                for layer in self._LAYERS:
                    forward_inputs = layer.forward(
                        forward_inputs,
                        in_training=True)
                batch_outputs = forward_inputs

                # Backward propagate through the model
                output_gradients = self._LOSS.gradate(
                    batch_outputs,
                    batch_labels)
                backward_gradients = output_gradients
                for layer in reversed(self._LAYERS):
                    backward_gradients = layer.backward(backward_gradients)

                # Update the layers with the parameter gradients
                for layer in self._LAYERS:
                    if layer.IS_TRAINABLE:
                        layer.update(optimizer, learning_rate)

            # Calculate and cache the current model metrics
            current_outputs = self.predict(examples)
            current_loss = self._LOSS.calculate(current_outputs, labels)
            training_history[epoch] = current_loss

            # Print the current model performance
            print(f"Epoch {epoch + 1}: loss {round(current_loss, 4)}")

        # Message that the model is trained
        print(f"Model is fit after {epochs} epochs\n")

        return training_history  # Return the training history
