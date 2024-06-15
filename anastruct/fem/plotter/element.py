import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from anastruct.fem.elements import Element


def plot_values_deflection(
    element: "Element", factor: float, linear: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Determines the plotting values for deflection

    Args:
        element (Element): Element to plot
        factor (float): Factor by which to multiply the plotting values perpendicular to the elements axis.
        linear (bool, optional): If True, the bending in between the elements is determined. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values
    """
    ux1 = element.node_1.ux * factor
    uy1 = -element.node_1.uy * factor
    ux2 = element.node_2.ux * factor
    uy2 = -element.node_2.uy * factor

    x1 = element.vertex_1.x + ux1
    y1 = element.vertex_1.y + uy1
    x2 = element.vertex_2.x + ux2
    y2 = element.vertex_2.y + uy2

    if element.type == "general" and not linear:
        assert element.deflection is not None
        n = len(element.deflection)
        x_val = np.linspace(x1, x2, n)
        y_val = np.linspace(y1, y2, n)

        x_val = x_val + element.deflection * math.sin(element.angle) * factor
        y_val = y_val + element.deflection * -math.cos(element.angle) * factor

    else:  # truss element has no bending
        x_val = np.array([x1, x2])
        y_val = np.array([y1, y2])

    return x_val, y_val


def plot_values_bending_moment(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Determines the plotting values for bending moment

    Args:
        element (Element): Element to plot
        factor (float): Factor by which to multiply the plotting values perpendicular to the elements axis.
        n (int): Number of points to plot

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values
    """

    # Determine forces for horizontal element.angle = 0
    T_left = element.node_1.Tz
    T_right = -element.node_2.Tz

    sin = math.sin(-element.angle)
    cos = math.cos(-element.angle)

    # apply angle ai
    x1 = element.vertex_1.x + T_left * sin * factor
    y1 = element.vertex_1.y + T_left * cos * factor
    x2 = element.vertex_2.x + T_right * sin * factor
    y2 = element.vertex_2.y + T_right * cos * factor

    interpolate = np.linspace(0, 1, n)
    dx = x2 - x1
    dy = y2 - y1

    # determine moment for 0 < x < length of the element
    x_val = x1 + interpolate * dx
    y_val = y1 + interpolate * dy

    if element.q_load or element.dead_load:
        qi = element.all_qp_load[0]
        q = element.all_qp_load[1]
        x = interpolate * element.l
        q_part = (
            -((qi - q) / (6 * element.l)) * x**3
            + (qi / 2) * x**2
            - (((2 * qi) + q) / 6) * element.l * x
        )
        x_val += sin * q_part * factor
        y_val += cos * q_part * factor

    x_val = np.append(x_val, element.vertex_2.x)
    y_val = np.append(y_val, element.vertex_2.y)
    x_val = np.insert(x_val, 0, element.vertex_1.x)
    y_val = np.insert(y_val, 0, element.vertex_1.y)

    return x_val, y_val


def plot_values_axial_force(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Determines the plotting values for axial force

    Args:
        element (Element): Element to plot
        factor (float): Factor by which to multiply the plotting values perpendicular to the elements axis.
        n (int): Number of points to plot

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values
    """

    # Determine forces for horizontal element.angle = 0
    assert element.N_1 is not None
    assert element.N_2 is not None
    N1 = element.N_1
    N2 = element.N_2

    cos = math.cos(0.5 * math.pi + element.angle)
    sin = math.sin(0.5 * math.pi + element.angle)

    # apply angle ai
    x1 = element.vertex_1.x + N1 * cos * factor
    y1 = element.vertex_1.y + N1 * sin * factor
    x2 = element.vertex_2.x + N2 * cos * factor
    y2 = element.vertex_2.y + N2 * sin * factor

    interpolate = np.linspace(0, 1, n)
    dx = x2 - x1
    dy = y2 - y1

    # determine axial force for 0 < x < length of the element
    x_val = x1 + interpolate * dx
    y_val = y1 + interpolate * dy

    if element.q_load or element.dead_load:
        qni = element.all_qn_load[0]
        qn = element.all_qn_load[1]
        x = interpolate * element.l
        qn_part = (qn - qni) / (2 * element.l) * (element.l - x) * x
        x_val += cos * qn_part * factor
        y_val += sin * qn_part * factor

    x_val = np.append(x_val, element.vertex_2.x)
    y_val = np.append(y_val, element.vertex_2.y)
    x_val = np.insert(x_val, 0, element.vertex_1.x)
    y_val = np.insert(y_val, 0, element.vertex_1.y)

    return x_val, y_val


def plot_values_shear_force(
    element: "Element", factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Determines the plotting values for shear force

    Args:
        element (Element): Element to plot
        factor (float): Factor by which to multiply the plotting values perpendicular to the elements axis.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values
    """
    x1 = element.vertex_1.x
    y1 = element.vertex_1.y
    x2 = element.vertex_2.x
    y2 = element.vertex_2.y

    assert element.shear_force is not None
    n = len(element.shear_force)

    # apply angle ai
    interpolate = np.linspace(0, 1, n)
    dx = x2 - x1
    dy = y2 - y1

    x_val = x1 + interpolate * dx
    y_val = y1 + interpolate * dy

    sin = math.sin(-element.angle)
    cos = math.cos(-element.angle)

    x_val += sin * element.shear_force * factor
    y_val += cos * element.shear_force * factor

    x_val = np.append(x_val, element.vertex_2.x)
    y_val = np.append(y_val, element.vertex_2.y)
    x_val = np.insert(x_val, 0, element.vertex_1.x)
    y_val = np.insert(y_val, 0, element.vertex_1.y)

    return x_val, y_val


def plot_values_element(element: "Element") -> Tuple[np.ndarray, np.ndarray]:
    """Determines the plotting values for the element itself (e.g. for plot_structure())

    Args:
        element (Element): Element to plot

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values
    """
    x_val = np.array([element.vertex_1.x, element.vertex_2.x])
    y_val = np.array([element.vertex_1.y, element.vertex_2.y])
    return x_val, y_val
