"""Virhefunktio joita käytetään minimoimisessa."""


def cross_entropy_cost_derivative(a, y):
    """
    Cross-Entropy virhefunktion derivaatta
    
    http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function

    Params:
    a (np-array): Ulos menevät aktivoinnit.
    y (np-array): Oikeat arvot (Labels).
    """

    # Käännetään y - a jotta saadaan vastaluku.
    # Tämä tehdään, koska gradienttimenetlmässä halutaan mennä
    # derivaatan vastakkaiseen suuntaan.
    return a - y
