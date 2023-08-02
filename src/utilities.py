"""Apufunktiota."""

def split_every(n, lst):
    """
    Jaetaan lista n kokoisiin alilistoihin.

    Args:
        n (int): Alilistan koko.
        lst (list): Jaettava lista.

    Palauttaa:
        list: lista alilistoja
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]
