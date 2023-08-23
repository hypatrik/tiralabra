"""Aktivointi funktiot ja niiden derivaatat."""

import numpy as np

# Funktioiden kuvaajat löytyvät activation_functions.ipynb notebookista.


class ReLU:
    """ReLU (Rectified Linear Unit)."""

    def activation(self, x):
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

    def derivative(self, x):
        """
        ReLu derivaatta.

        Args:
        x (numpy array)

        Returns:
            numpy array
        """
        return np.where(x > 0, 1, 0)


class Tanh:
    """Tanh."""

    def __init__(self, alpha=1.0):
        """Konstruktori.

        Args:
            alpha (float, optional): Skaalauskerroin. Defaults to 1.
        """
        self.alpha = alpha

    def activation(self, x):
        """Tanh funktio.
        
        Tanh antaa arvoja välillä -1 ja 1. Se muistuttaa muodoltaan Sigmoidia.

        Args:
            x (np.array)

        Returns:
            numpy array
        """
        return np.tanh(self.alpha * x)

    def derivative(self, x):
        """Tanh funktion derivaatta.

        Args:
            x (np.array)

        Returns:
            numpy array
        """
        return 1 - np.tanh(self.alpha * x) ** 2


class Sigmoid:
    """Sigmoid."""

    def activation(self, x):
        """
        Sigmoid aktivointifunktio.

        Sigmoid-funktio on yleisesti käytetty aktivointifunktio, joka
        palauttaa arvoja välillä [0,1]. Funktion kuvaaja on pehmeä transitio
        0:sta 1:teen.

        Args:
            x (numpy array)
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        Sigmoid-funktion derivaatta.

        Args:
            x (numpy array)
        """
        v = self.activation(x)
        return v * (1 - v)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activation(self, x):
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

        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        """
        Leaky ReLU aktivointifunktio.


        Args:
            x (numpy array)
            alpha (float): Negatiivisten arvojen kulmakerroin. Default 0.01.
        """

        return np.where(x > 0, 1, self.alpha)
