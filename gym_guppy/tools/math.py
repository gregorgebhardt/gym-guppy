import numpy as np


def normalize(x: np.ndarray):
    return x / np.sqrt((x ** 2).sum())


def rotation(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array(((c, -s), (s, c)))


def is_point_left(a, b, c):
    """computes if c is left of the line ab.
    :return: True if c is left of the line ab and False otherwise.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0