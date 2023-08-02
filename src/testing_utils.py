"""Testauksessa käytettäviä apufunktiota."""

import numpy as np

def assertAlmostEqual(x, y, digits=7):
    """
    Testaa onko kaksi numeerista arvoa liittävän lähellä toisiaan.

    Args:
        x (numeric / numpy lista)
        y (numeric / numpy list)
        digits (int): Monenko desimaalin tarkkuudella. Default: 7
    """
    multiplier = 10 ** digits
    assert np.round(x * multiplier) == np.round(y * multiplier), f"{x} and {y} are not almost equal."
