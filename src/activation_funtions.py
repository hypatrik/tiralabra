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
    
    Returns:
        numpy array
    """
    return np.where(x >= threshold, 1, 0)


def relu(x):
    """
    Relu aktivointifunktio.

    ReLU (Rectified Linear Unit) on yleisesti käytetty aktivointifunktio,
    joka palauttaa palauttaa vain posiitivisia arvoja. Mikäli annettu arvo on
    negatiivinen, palautetaan 0.
    
    https://tim.jyu.fi/view/143092#relu

    Args:
        x (numpy array)
    
    Returns:
        numpy array
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    ReLu derivaatta.

    Args:
    x (numpy array)
    
    Returns:
        numpy array
    """
    return np.where(x > 0, 1, 0)

def tanh(x, alpha=1.0):
    """Tanh funktio.

    Args:
        x (np.array)
        alpha (float, optional): Skaalauskerroin. Defaults to 1.

    Returns:
        numpy array
    """
    return np.tanh(alpha * x)

def tanh_derivative(x, alpha=1.0):
    """Tanh funktion derivaatta.

    Args:
        x (np.array)
        alpha (float, optional): Skaalauskerroin. Defaults to 1.

    Returns:
        numpy array
    """
    return 1 - np.tanh(alpha * x) ** 2

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

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU aktivointifunktio.
    
    ReLu-funktio ei ole derivoituva nollassa. Sen toinen huono ominaisuus on se, että se on nolla
    ja sen derivaatta on nolla negatiivisilla arvoilla. Tästä syystä joidenkin neuronien painot
    saattavat päivittyä oppimisen aikana nollaksi jolloin neuronit kuolevat. Neuronien kuoleentumisongelmaa
    pyritään välttämään muuttamaan funktiota siten, että negatiivisilla arvoilla kulmakerroin on
    merkittävästi loivempi.
    
    Args:
        x (numpy array)
        alpha (float): Negatiivisten arvojen kulmakerroin. Default 0.01.
    """

    return np.maximum(alpha*x, x)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Leaky ReLU aktivointifunktio.
    

    Args:
        x (numpy array)
        alpha (float): Negatiivisten arvojen kulmakerroin. Default 0.01.
    """
    
    return np.where(x > 0, 1, alpha)

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
