import numpy as np
import math


def find_nearest(array, value):
    """
    :param array: (numpy array object)
    :param value: (float) value searched for
    :return: (tuple) nearest value, index
    """
    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index = (np.abs(array-value)).argmin()
    return array[index], index


def converge(lhs, rhs, div=3):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    return (abs(rhs) / abs(lhs) - 1) / div + 1


def angle_x_axis(delta_x, delta_z):
    # dot product v_x = [1, 0] ; v = [delta_x, delta_z]
    # dot product = 1 * delta_x + 0 * delta_z -> delta_x
    ai = math.acos(delta_x / math.sqrt(delta_x**2 + delta_z**2))
    if delta_z < 0:
        ai = 2 * math.pi - ai

    return ai


def integrate_array(y, dx):
    """
    integrate array y * dx
    """
    y_int = np.zeros(y.size)
    for i in range(y.size - 1):
        y_int[i + 1] = y_int[i] + y[i + 1] * dx
    return y_int


class FEMException(Exception):
    def __init__(self, type_, message):
        self.type = type_
        self.message = message

