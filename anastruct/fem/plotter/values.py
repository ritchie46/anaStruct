import math
import numpy as np
from .element import (
    plot_values_deflection,
    plot_values_bending_moment,
    plot_values_axial_force,
    plot_values_shear_force,
    plot_values_element,
)


def det_scaling_factor(max_unit, max_val_structure):
    if math.isclose(max_unit, 0):
        return 0.1
    else:
        return 0.15 * max_val_structure / max_unit


class PlottingValues:
    def __init__(self, system, mesh):
        self.system = system
        self.mesh = max(3, mesh)
        # used for scaling the plotting values.
        self._max_val_structure = None

    @property
    def max_val_structure(self):
        """
        Determine the maximum value of the structures dimensions.
        :return: (flt)
        """
        if self._max_val_structure is None:
            self._max_val_structure = max(
                map(
                    lambda el: max(
                        el.vertex_1.x, el.vertex_2.x, el.vertex_1.y, el.vertex_2.y
                    ),
                    self.system.element_map.values(),
                )
            )
        return self._max_val_structure

    def displacements(self, factor, linear):
        if factor is None:
            # needed to determine the scaling factor
            max_displacement = max(
                map(
                    lambda el: max(
                        abs(el.node_1.ux), abs(el.node_1.uz), el.max_deflection
                    )
                    if el.type == "general"
                    else 0,
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_displacement, self.max_val_structure)
        xy = np.hstack(
            [
                plot_values_deflection(el, factor, linear)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def bending_moment(self, factor):
        if factor is None:
            # maximum moment determined by comparing the node's moments and the sagging moments.
            max_moment = max(
                map(
                    lambda el: max(
                        abs(el.node_1.Ty),
                        abs(el.node_2.Ty),
                        abs(((el.all_qp_load[0] + el.all_qp_load[1]) / 16) * el.l**2),
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_moment, self.max_val_structure)

        n = len(self.system.element_map[1].bending_moment)
        xy = np.hstack(
            [
                plot_values_bending_moment(el, factor, n)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def axial_force(self, factor):
        if factor is None:
            max_force = max(
                map(
                    lambda el: max(
                        abs(el.N_1),
                        abs(el.N_2),
                        abs(((el.all_qn_load[0] + el.all_qn_load[1]) / 2) * el.l),
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)
        n = len(self.system.element_map[1].axial_force)
        xy = np.hstack(
            [
                plot_values_axial_force(el, factor, n)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def shear_force(self, factor):
        if factor is None:
            max_force = max(
                map(
                    lambda el: np.max(np.abs(el.shear_force)),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)
        xy = np.hstack(
            [
                plot_values_shear_force(el, factor)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def structure(self):
        xy = np.hstack(
            [plot_values_element(el) for el in self.system.element_map.values()]
        )
        return xy[0, :], xy[1, :]
