from .element import plot_values_deflection
import numpy as np
import math


def det_scaling_factor(max_unit, max_val_structure):
    if math.isclose(max_unit, 0):
        return 0.1
    else:
        return 0.15 * max_val_structure / max_unit


class PlottingValues:
    def __init__(self, system, mesh):
        self.system = system
        self.mesh = mesh
        # used for scaling the plotting values.
        self.max_val_structure = max(map(lambda el: max(el.vertex_1.x, el.vertex_2.x, el.vertex_1.y, el.vertex_2.y),
                                         self.system.element_map.values()))

    def displacements(self, factor, linear):
        if factor is None:
            # needed to determine the scaling factor
            max_displacement = max(map(lambda el: max(abs(el.node_1.ux), abs(el.node_1.uz), el.max_deflection),
                                   self.system.element_map.values()))
            factor = det_scaling_factor(max_displacement, self.max_val_structure)
        xy = np.hstack([plot_values_deflection(el, factor, linear) for el in self.system.element_map.values()])
        return xy[0, :], xy[1, :]

