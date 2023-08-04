import numpy as np
from activation_funtions import activation_function_factory, sigmoid

from sgd import stochastic_gradient_descent_fn, update_fn_factory
from backpropagation import backpropagation_fn_factory
from utilities import calculate_z, init_weights_and_biases


class NeuralNetwork:
    def __init__(self, layers, activation_function="sigmoid") -> None:
        self.layers = layers
        weights, biases = init_weights_and_biases(layers)
        self.weights = weights
        self.biases = biases
        af, afd = activation_function_factory(activation_function)
        backpropagation_fn = backpropagation_fn_factory(
            activation_function=af, activation_function_derivative=afd
        )
        self.sgd = stochastic_gradient_descent_fn(
            update_fn=update_fn_factory(backpropagation_fn=backpropagation_fn)
        )

    def fit(self, X, y, epochs=30, learning_rate=100, batch_size=100):
        w, b = self.sgd(
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
        prediction = np.argmax(y)
        confidence = np.round(y, 3)[prediction]

        return prediction, confidence[0], y

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(calculate_z(a, w, b))
        return a
