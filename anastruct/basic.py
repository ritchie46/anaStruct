import collections.abc
from typing import Any, Tuple

import numpy as np

try:
    from anastruct.cython.cbasic import (  # pylint: disable=unused-import
        angle_x_axis,
        converge,
    )
except ImportError:
    from anastruct.cython.basic import angle_x_axis, converge


def find_nearest(array: np.ndarray, value: float) -> Tuple[float, int]:
    """
    :param array: (numpy array object)
    :param value: (float) value searched for
    :return: (tuple) nearest value, index
    """
    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index = (np.abs(array - value)).argmin()
    return array[index], index


def integrate_array(y: np.ndarray, dx: float) -> np.ndarray:
    """
    integrate array y * dx
    """
    return np.cumsum(y) * dx  # type: ignore


class FEMException(Exception):
    def __init__(self, type_: str, message: str):
        self.type = type_
        self.message = message


def arg_to_list(arg: Any, n: int) -> list:
    if isinstance(arg, list) and not isinstance(arg, str) and len(arg) == n:
        return arg
    else:
        return [arg for _ in range(n)]


def args_to_lists(*args: list) -> list:
    arg_lists = []
    for arg in args:
        if isinstance(arg, collections.abc.Iterable) and not isinstance(arg, str):
            arg_lists.append(arg)
        else:
            arg_lists.append([arg])
    lengths = list(map(len, arg_lists))
    n = max(lengths)
    if n == 1:
        return arg_lists

    args_return = []
    for arg, l in zip(arg_lists, lengths):
        if l == n:
            args_return.append(arg)
        else:
            args_return.append([arg[0] for _ in range(n)])
    return args_return


def rotation_matrix(angle: float) -> np.ndarray:
    """

    :param angle: (flt) angle in radians
    :return: rotated euclidean xy matrix
    """
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s], [s, c]])


def rotate_xy(a: np.ndarray, angle: float) -> np.ndarray:
    b = np.array(a)
    b[:, 0] -= a[0, 0]
    b[:, 1] -= a[0, 1]
    b = np.dot(b, rotation_matrix(angle))
    b[:, 0] += a[0, 0]
    b[:, 1] += a[0, 1]
    return b
