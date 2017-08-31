import math

cpdef converge(double lhs, double rhs, int div=3):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    return (abs(rhs) / abs(lhs) - 1) / div + 1

cpdef angle_x_axis(double delta_x, double delta_z):
    cdef double ai
    # dot product v_x = [1, 0] ; v = [delta_x, delta_z]
    # dot product = 1 * delta_x + 0 * delta_z -> delta_x
    ai = math.acos(delta_x / math.sqrt(delta_x**2 + delta_z**2))
    if delta_z < 0:
        ai = 2 * math.pi - ai

    return ai