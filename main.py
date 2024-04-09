"""
Build a machine learning model to recognize handwritten digits.
"""

__author__ = 'Dylan Warnecke'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from framework.activations import Relu, Tanh
from framework.layers import Dense, Dropout, Input
from framework.layers import Convolution, Flatten, MaxPool
from framework.losses.categorical_cross_entropy import CategoricalCrossEntropy
from framework.model import Model
from framework.optimizers.schedules import Exponential
from framework.optimizers import Adam
from framework.utilities import make_one_hot


if __name__ == '__main__':
    # Import and convert the training data
    training_set = pd.read_csv('training-digits.csv')
    training_pixels = training_set.drop(columns='label').to_numpy()
    training_digits = training_set.loc[:, 'label'].to_numpy()
    training_digits = make_one_hot(training_digits, 10)
    training_images = training_pixels.reshape((-1, 28, 28, 1))

    # Make the dense network model
    model = Model(
        CategoricalCrossEntropy(from_logits=True),
        Input((28, 28, 1)),
        Convolution(16, 3, True, Relu()),
        Dropout(0.1),
        MaxPool(2),
        Convolution(64, 3, True, Relu()),
        Dropout(0.1),
        MaxPool(2),
        Flatten(),
        Dense(256, Tanh()),
        Dropout(0.5),
        Dense(10))

    # Fit the model using gradient descent
    training_history = model.fit(
        training_images,
        training_digits,
        batch_size=2**9,
        epochs=2**5,
        optimizer=Adam(0.9, 0.99, Exponential(2e-4, 0.98)))

    # Plot the model training history
    fig = plt.figure(figsize=(4, 4))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(training_history)
    axes.set_title("Training Model Loss Over Epochs")
    axes.set_xlabel("Epoch")
    axes.set_xscale('log')
    axes.set_ylabel("Loss")
    axes.set_yscale('log')
    plt.show(block=True)

    # Save the model for later importation
    digit_model.save('model.json')
