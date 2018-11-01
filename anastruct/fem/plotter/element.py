import numpy as np
import math


def plot_values_deflection(element, factor, linear=False):
    """
    Determine the plotting values for deflection

    :param element: (fem.Element)
    :param factor: (flt) Factor by which to multiply the plotting values perpendicular to the elements axis.
    :param linear: (bool) If True, the bending in between the elements is determined.
    :return: (np.array/ list) x and y values.
    """
    ux1 = element.node_1.ux * factor
    uz1 = -element.node_1.uz * factor
    ux2 = element.node_2.ux * factor
    uz2 = -element.node_2.uz * factor

    x1 = element.vertex_1.x + ux1
    y1 = -element.vertex_1.z + uz1
    x2 = element.vertex_2.x + ux2
    y2 = -element.vertex_2.z + uz2

    if element.type == "general" and not linear:
        n = len(element.deflection)
        x_val = np.linspace(x1, x2, n)
        y_val = np.linspace(y1, y2, n)

        x_val = x_val + element.deflection * math.sin(element.ai) * factor
        y_val = y_val + element.deflection * -math.cos(element.ai) * factor

    else:  # truss element has no bending
        x_val = np.array([x1, x2])
        y_val = np.array([y1, y2])

    return x_val, y_val
