import random
import numpy as np
from activation_funtions import sigmoid

from sgd import stochastic_gradient_descent
from utilities import calculate_z, init_weights_and_biases, zero_weight_and_bias_vectors


class NeuralNetwork:
    def __init__(self, layers) -> None:
        self.layers = layers
        weights, biases = init_weights_and_biases(layers)
        self.weights = weights
        self.biases = biases

    def fit(self, X, y, epochs=30, learning_rate=100, batch_size=100):
        w, b = stochastic_gradient_descent(
            self.weights,
            self.biases,
            X,
            y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        
        self.weights = w
        self.biases = b

    def predict(self, x):
        y = self.feedforward(x)
        return np.argmax(y), y

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(calculate_z(a, w, b))
        return a
