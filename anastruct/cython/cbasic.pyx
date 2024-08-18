from libc.stdlib cimport abs
from libc.math cimport acos, sqrt, pi, abs

cpdef converge(double lhs, double rhs):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    lhs = abs(lhs)
    rhs = abs(rhs)
    # cdef double div
    div = max(lhs, rhs) / min(lhs, rhs) * 2.0
    return (abs(rhs) / abs(lhs) - 1.0) / div + 1.0

cpdef angle_x_axis(double delta_x, double delta_y):
    cdef double ai
    # dot product v_x = [1, 0] ; v = [delta_x, delta_y]
    # dot product = 1 * delta_x + 0 * delta_y -> delta_x
    ai = acos(delta_x / sqrt(delta_x**2 + delta_y**2))
    if delta_y < 0:
        ai = 2 * pi - ai

    return ai