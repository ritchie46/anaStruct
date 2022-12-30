import math


def converge(lhs, rhs):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    lhs = abs(lhs)
    rhs = abs(rhs)
    div = max(lhs, rhs) / min(lhs, rhs) * 2

    return (rhs / lhs - 1) / div + 1


def angle_x_axis(delta_x, delta_z):
    # dot product v_x = [1, 0] ; v = [delta_x, delta_z]
    # dot product = 1 * delta_x + 0 * delta_z -> delta_x
    ai = math.acos(delta_x / math.sqrt(delta_x**2 + delta_z**2))
    if delta_z < 0:
        ai = 2 * math.pi - ai

    return ai
