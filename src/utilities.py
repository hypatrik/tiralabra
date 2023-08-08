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
    """Apufunktio nolla matriisien luomiseen.

    Luodaan saman kokoiset nolla matriisit kuin weights ja biases.

    Args:
        weights (np.array): Painot
        biases (np.array): Vakiot

    Returns:
        np.array: Painojen nolla matriisi
        np.array: Vakioiden nolla matriisi
    """
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

    # Huomasin, että saan parempi tuloksia, kun alustan verkon skaalatulla
    # normaalijakuamalla. Tässä normaaljakaumsta saadut satunnaislukuvektorit kerrotaan
    # np.sqrt(2.0) / n_neurons. Esimerkiksi Leaky ReLU funktio ei juuri toiminus ilman tätä.
    biases = [np.random.randn(n_neurons, 1) * np.sqrt(2.0) / n_neurons for n_neurons in layers[1:]]
    weights = [
        np.random.randn(n_neurons, n_neurons_prev_layer) * np.sqrt(2.0) / n_neurons
        for n_neurons, n_neurons_prev_layer in zip(layers[1:], layers[:-1])
    ]

    return weights, biases


def calculate_z(a, w, b):
    """Laskee painoitetun syöteen kerrokselle l.

    z^l≡w^l * a^l-1 + b^l

    Args:
        a (_type_): Aktivointivektori kerrokselle l-1
        w (_type_): Painovektori kerrokselle l
        b (_type_): Vakiovektori kerrokselle l

    Returns:
        np.array: Painoitettu syöte kerrokselle l
    """
    return np.dot(w, a) + b


class InvalidNumericInputException(Exception):
    """Validointi virhe numerisille arvoille."""

    pass


def vectorize_label(number):
    """Apufunktio vektorisoida kokonaisluku 0-9.

    Args:
        number (int): 0-9.

    Returns:
        np.array: Vektori 10x1
    """

    if number < 0 or number > 9:
        raise InvalidNumericInputException("Number should be between 0 and 9.")

    y = np.zeros((10, 1))
    y[number] = 1.0
    return y
