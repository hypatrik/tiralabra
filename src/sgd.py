import random
import numpy as np

from utilities import split_every, zero_weight_and_bias_vectors
from backpropagation import backpropagation

def update(weights, biases, batch, learning_rate, backpropagation_fn=backpropagation):
        weight_gradients, bias_gradients = zero_weight_and_bias_vectors(weights, biases)
        
        batch_size = len(batch)

        for x, y in batch:
            delta_weights, delta_biases = backpropagation_fn(weights, biases, x, y)
                
            weight_gradients = [
                w_gradient + delta_w
                for w_gradient, delta_w in zip(weight_gradients, delta_weights)
            ]
            bias_gradients = [
                b_gradient + delta_b
                for b_gradient, delta_b in zip(bias_gradients, delta_biases)
            ]

        weights = [
            w - (learning_rate / batch_size) * w_gradient
            for w, w_gradient in zip(weights, weight_gradients)
        ]
        
        biases = [
            b - (learning_rate / batch_size) * b_gradient
            for b, b_gradient in zip(biases, bias_gradients)
        ]
        
        # Palautetaan testaamista varten
        # Python funktion otsakkeen muuttujat ovat aina viittauksia.
        return weights, biases


def stochastic_gradient_descent(
    weights, biases, X, y, epochs=30, learning_rate=10, batch_size=100, update_fn=update
):
    training_data = list(zip(X, y))

    for i in range(epochs):
        random.shuffle(training_data)
        for batch in split_every(batch_size, training_data):
            weights, biases = update_fn(weights, biases, batch, learning_rate)
        print("epoch {} done".format(i))

    return weights, biases