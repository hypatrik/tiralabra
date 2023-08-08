"""Aktivointi funktiot ja niiden derivaatat."""

import numpy as np


class NoSuchActivationFunctionException(Exception):
    """Virhe mikäli aktivointi funktiota ei ole määritelty."""


# Funktioiden kuvaajat löytyvät activation_functions.ipynb notebookista.


def step(x, threshold=0):
    """
    Step aktivointifunktio.

    Käyttämällä step-funktiota neuroverkon neuronit ovat
    perseptroneja (perceptron), eli binäärisiä 0 tai 1 arvoja.
    Perseptroneja käytettiin neuroverkoissa etenkin 1950 ja 1960 luvuilla.

    Args:
        x (numpy array)
        threshold (float): Jos x[i] on suurempi tai yhtä kuin threshold
                    palautetaan 1
    """
    return np.where(x >= threshold, 1, 0)


def relu(x):
    """
    Relu aktivointifunktio.

    ReLU (Rectified Linear Unit) on yleisesti käytetty aktivointifunktio,
    joka palauttaa palauttaa vain posiitivisia arvoja. Mikäli annettu arvo on
    negatiivinen, palautetaan 0.

    Args:
        x (numpy array)
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid aktivointifunktio.

    Sigmoid-funktio on yleisesti käytetty aktivointifunktio, joka
    palauttaa arvoja välillä [0,1]. Funktion kuvaaja on pehmeä transitio
    0:sta 1:teen.

    Args:
        x (numpy array)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid-funktion derivaatta.

    Args:
        x (numpy array)
    """
    v = sigmoid(x)
    return v * (1 - v)


def activation_function_factory(function_name):
    """
    Palauttaa halutun aktivointi funktiot ja sen derivaatan.

    Nostaa NoSuchActivationFunctionException virheen mikäli aktivointfunktiota
    ei ole määritelty.

    Args:
        function_name (string): sigmoid, relu, step
    """

    try:
        f = globals()[function_name]
        fd = globals()["{}_derivative".format(function_name)]

        return f, fd
    except:
        raise NoSuchActivationFunctionException("No such activation function")
