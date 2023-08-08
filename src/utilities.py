"""Apufunktiota."""

import numpy as np


def split_every(n, lst):
    """
    Jaetaan lista n kokoisiin alilistoihin.

    Args:
        n (int): Alilistan koko.
        lst (list): Jaettava lista.

    Palauttaa:
        list: lista alilistoja
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def zero_weight_and_bias_vectors(weights, biases):
    return (
        [np.zeros_like(w) for w in weights],
        [np.zeros_like(b) for b in biases],
    )

def init_weights_and_biases(layers):
    """
    Alustetaan painot (weight) ja vakiotermit (bias).

    Arvoista on otettu pois syötekerros. Myöhemmin tämä helpottaa muun muassa
    vastavirta-argoritmissä (backpropagation).

    Args:
        layers (list)
    """
    biases = [np.random.randn(n_neurons, 1) for n_neurons in layers[1:]]
    weights = [
        np.random.randn(n_neurons, n_neurons_prev_layer)
        for n_neurons, n_neurons_prev_layer in zip(layers[1:], layers[:-1])
    ]
    
    return weights, biases


def calculate_z(a, w, b):
    return np.dot(w, a) + b

def vectorize_label(number):
    y = np.zeros((10, 1))
    y[number] = 1.0
    return y
