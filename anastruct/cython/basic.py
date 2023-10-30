import math


def converge(lhs: float, rhs: float) -> float:
    """Determine convergence factor

    Args:
        lhs (float): The left-hand side of the equation
        rhs (float): The right-hand side of the equation

    Returns:
        float: Convergence factor ((lhs / rhs) - 1) / div + 1
    """
    lhs = abs(lhs)
    rhs = abs(rhs)
    div = max(lhs, rhs) / min(lhs, rhs) * 2

    return (rhs / lhs - 1) / div + 1


def angle_x_axis(delta_x: float, delta_y: float) -> float:
    """Determine the angle of the element with the global x-axis

    Args:
        delta_x (float): Element length in the x-direction
        delta_y (float): Element length in the y-direction

    Returns:
        float: Angle of the element with the global x-axis
    """
    # dot product v_x = [1, 0] ; v = [delta_x, delta_y]
    # dot product = 1 * delta_x + 0 * delta_y -> delta_x
    ai = math.acos(delta_x / math.sqrt(delta_x**2 + delta_y**2))
    if delta_y < 0:
        ai = 2 * math.pi - ai

    return ai
