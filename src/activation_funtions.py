import numpy as np

# Funktioiden kuvaajat löytyvät activation_functions.ipynb notebookista.

def step(x, threshold = 0):
    """
    Käyttämällä step-funktiota neuroverkon neuronit ovat
    perseptroneja (perceptron), eli binäärisiä 0 tai 1 arvoja.
    Perseptroneja käytettiin neuroverkoissa etenkin 1950 ja 1960 luvuilla.

    Parametrit:
    x:          numpy array
    threshold:  numeric. Jos x[i] on suurempi tai yhtä kuin threshold
                palautetaan 1
    """
    return np.where(x >= threshold, 1, 0)

def relu(x):
    """
    ReLU (Rectified Linear Unit) on yleisesti käytetty aktivointifunktio,
    joka palauttaa palauttaa vain posiitivisia arvoja. Mikäli annettu arvo on
    negatiivinen, palautetaan 0.

    Parametrit:
    x: numpy array
    """
    return np.maximum(0, x)

def sigmoid(x):
    """
    Sigmoid-funktio on yleisesti käytetty aktivointifunktio, joka
    palauttaa arvoja välillä [0,1]. Funktion kuvaaja on pehmeä transitio
    0:sta 1:teen.

    Parametrit:
    x: numpy array
    """
    return 1 / (1 + np.exp(-x))

def activation_function_factory(function_name):
    f = globals()[function_name]
    if not f:
        raise 'No such activation function'
    return f