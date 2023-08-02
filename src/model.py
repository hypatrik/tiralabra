"""Neuroverkon toteutus."""

import numpy as np

from activation_funtions import activation_function_factory


class NeuralNetwork:
    """Neuroverkko."""

    def __init__(self, layers, activation_function="sigmoid"):
        """
        _summary_.

        Args:
            layers (_type_): _description_
            activation_function (str, optional): _description_. Defaults to "sigmoid".
        """
        self.n_layers = len(layers)

        self.biases = [np.random.rand(n_neurons, 1) for n_neurons in layers]
        self.weights = [
            np.random.rand(n_neurons, n_neurons_prev_layer)
            for n_neurons, n_neurons_prev_layer in zip(layers[:1], layers[:-1])
        ]

        af, adf = activation_function_factory(activation_function)
        self.activation_function = af
        self.activation_function_derivative = adf

    def fit(self, X, y):
        """
        _summary_.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        pass