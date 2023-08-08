"""Virhefunktio joita käytetään minimoimisessa."""


def quadratic_cost_function_derivative(a, y):
    """
    Neliöllisen virhefunktion (Mean squared error) osittaisderivaatan vastaluku.

    MSE => C = 1/2 * ||y - a||²
    ja derivaatta on
    ∇yC = y - a

    Params:
    a (np-array): Ulos menevät aktivoinnit.
    y (np-array): Oikeat arvot (Labels).
    """

    # Käännetään y - a jotta saadaan vastaluku.
    # Tämä tehdään, koska gradienttimenetlmässä halutaan mennä
    # derivaatan vastakkaiseen suuntaan.
    return a - y
