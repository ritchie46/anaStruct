import copy
import math
from typing import TYPE_CHECKING
import numpy as np
from anastruct.fem.node import Node
from anastruct.basic import integrate_array


if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements
    from anastruct.fem.elements import Element


class SystemLevel:
    def __init__(self, system: "SystemElements"):
        self.system = system
        # post processor element level
        self.post_el = ElementLevel(self.system)

    def node_results_elements(self):
        """
        Determines the node results on the system level.
        Results placed in SystemElements class: self.node_objects (list).
        """

        for el in self.system.element_map.values():
            # post processor element level
            self.post_el.node_results(el)

    def node_results_system(self):
        for k, v in self.system.node_element_map.items():
            # reset nodes in case of iterative calculation
            self.system.node_map[k].reset()

            if k in self.system.loads_moment:
                self.system.node_map[k].Ty += self.system.loads_moment[k]

            if k in self.system.loads_point:
                Fx, Fz = self.system.loads_point[k]
                self.system.node_map[k].Fx += Fx
                self.system.node_map[k].Fz += Fz

            for vi in v:
                node = vi.node_map[k]
                self.system.node_map[k] -= node

                # The displacements are not summarized. Should be assigned only once
                self.system.node_map[k].ux = -node.ux
                self.system.node_map[k].uz = -node.uz
                self.system.node_map[k].phi_y = -node.phi_y

    def reaction_forces(self):
        supports = []
        for node in self.system.supports_fixed:
            supports.append(node.id)
        for node in self.system.supports_hinged:
            supports.append(node.id)
        for node in self.system.supports_roll:
            supports.append(node.id)
        for node in self.system.supports_rotational:
            supports.append(node.id)
        for node, _ in self.system.supports_spring_x:
            supports.append(node.id)
        for node, _ in self.system.supports_spring_z:
            supports.append(node.id)
        for node, _ in self.system.supports_spring_y:
            supports.append(node.id)

        for node_id in supports:
            node = self.system.node_map[node_id]
            node = copy.copy(node)
            self.system.reaction_forces[node_id] = node
            node.Fx *= -1
            node.Fz *= -1
            node.Ty *= -1
            node.ux = None
            node.uz = None
            node.phi_y = None

    def element_results(self):
        """
        Determines the element results for all elements in the system on element level.
        """

        for el in self.system.element_map.values():
            con = self.system.plotter.mesh
            self.post_el.determine_axial_force(el, con)
            self.post_el.determine_bending_moment(el, con)
            self.post_el.determine_shear_force(el, con)
            self.post_el.determine_displacements(el, con)


class ElementLevel:
    def __init__(self, system: "SystemElements"):
        self.system = system

    def node_results(self, element: "Element"):
        """
        Determine node results on the element level.
        """
        assert element.element_force_vector is not None
        assert element.element_primary_force_vector is not None

        # Check for hinges
        hinge1 = self.system.node_map[element.node_id1].hinge
        hinge2 = self.system.node_map[element.node_id2].hinge

        # Global coordinates system
        element.node_map[element.node_id1] = Node(
            id=element.node_id1,
            Fx=element.element_force_vector[0]
            + element.element_primary_force_vector[0],
            Fz=element.element_force_vector[1]
            + element.element_primary_force_vector[1],
            Ty=element.element_force_vector[2] + element.element_primary_force_vector[2]
            if not hinge1
            else 0,
            ux=element.element_displacement_vector[0],
            uz=element.element_displacement_vector[1],
            phi_y=element.element_displacement_vector[2] if not hinge1 else 0,
            hinge=hinge1,
        )

        element.node_map[element.node_id2] = Node(
            id=element.node_id2,
            Fx=element.element_force_vector[3]
            + element.element_primary_force_vector[3],
            Fz=element.element_force_vector[4]
            + element.element_primary_force_vector[4],
            Ty=element.element_force_vector[5] + element.element_primary_force_vector[5]
            if not hinge2
            else 0,
            ux=element.element_displacement_vector[3],
            uz=element.element_displacement_vector[4],
            phi_y=element.element_displacement_vector[5] if not hinge2 else 0,
            hinge=hinge2,
        )

        # Local coordinate system. With inclined supports
        for i in range(1, 3):
            a_n = getattr(element, f"a{i}")
            if a_n != element.angle:
                node = element.node_map[getattr(element, f"node_id{i}")]
                angle = a_n - element.angle
                c = np.cos(angle)
                s = np.sin(angle)
                Fx = node.Fx
                Fz = node.Fz
                ux = node.ux
                uz = node.uz
                node.Fz = c * Fz + s * Fx
                node.Fx = -(c * Fx + s * Fz)
                node.ux = c * ux + s * uz
                node.uz = c * uz + s * ux

    @staticmethod
    def determine_axial_force(element: "Element", con: int):
        N_1 = (math.sin(element.angle) * element.node_1.Fz) + -(
            math.cos(element.angle) * element.node_1.Fx
        )
        N_2 = -(math.sin(element.angle) * element.node_2.Fz) + (
            math.cos(element.angle) * element.node_2.Fx
        )

        element.N_1 = N_1
        element.N_2 = N_2

        dN = N_2 - N_1

        iteration_factor = np.linspace(0, 1, con)
        x = iteration_factor * element.l
        n_val = N_1 + iteration_factor * dN
        if element.all_qn_load:
            qni = element.all_qn_load[0]
            qn = element.all_qn_load[1]
            qn_part = (qn - qni) / (2 * element.l) * (element.l - x) * x
            n_val += qn_part

        element.axial_force = n_val

    @staticmethod
    def determine_bending_moment(element: "Element", con: int):
        dT = -(element.node_2.Ty + element.node_1.Ty)  # T2 - (-T1)

        iteration_factor = np.linspace(0, 1, con)
        x = iteration_factor * element.l
        m_val = element.node_1.Ty + iteration_factor * dT
        if element.all_qp_load:
            qi = element.all_qp_load[0]
            q = element.all_qp_load[1]
            q_part = (
                -((qi - q) / (6 * element.l)) * x**3
                + (qi / 2) * x**2
                - (((2 * qi) + q) / 6) * element.l * x
            )
            m_val += q_part

        element.bending_moment = m_val

    @staticmethod
    def determine_shear_force(element: "Element", con: int):
        """
        Determines the shear force by differentiating the bending moment.
        :param element: (object) of the Element class
        """
        iteration_factor = np.linspace(0, 1, con)
        x = iteration_factor * element.l
        eq = np.polyfit(x, element.bending_moment, 3)
        shear_force = eq[0] * 3 * x**2 + eq[1] * 2 * x + eq[2]
        element.shear_force = shear_force

    @staticmethod
    def determine_displacements(element: "Element", con: int):
        """
        Determines the displacement by integrating the bending moment.
        :param element: (object) of the Element class

        w = -M''

        This gives you the formula

        w = -aMx +bx + c

        a = already defined by the integral
        b = Scale the slope of the parabola. This is the rotation of the deflection.
            You can think of this as the angle of the deflection beam. By rotating
            the beam so that the last deflection w = 0 you get the correct
            value for b. w[-1] = 0.
        c = Translate the parabola. Translate it so that w[0] = 0
        """

        if element.type == "general":
            assert element.bending_moment is not None
            dx = element.l / (len(element.bending_moment) - 1)
            lx = np.linspace(0, element.l, con)

            # Next we are going to compute w by integrating from both sides.
            # Due to numerical differences we need to take this two sided approach.
            phi_neg1 = -integrate_array(element.bending_moment, dx) / element.EI
            w1 = integrate_array(phi_neg1, dx)

            # Angle between last w and elements axis. The w array will be corrected so that
            # this angle == 0.
            alpha1 = np.arctan(w1[-1] / element.l)
            w1 = w1 - lx * np.tan(alpha1)

            phi_neg2 = -integrate_array(element.bending_moment[::-1], dx) / element.EI
            w2 = integrate_array(phi_neg2, dx)

            # Angle between last w and elements axis. The w array will be corrected so that
            # this angle == 0.
            alpha2 = np.arctan(w2[-1] / element.l)
            w2 = w2[::-1] - lx[::-1] * np.tan(alpha2)

            element.deflection = -(w1 + w2) / 2.0
            element.max_deflection = np.max(np.abs(element.deflection))

        # Extension
        assert element.axial_force is not None
        dx = element.l / (len(element.axial_force) - 1)
        lx = np.linspace(0, element.l, con)

        # Next we are going to compute w by integrating from both sides.
        # Due to numerical differences we need to take this two sided approach.
        phi_neg1 = -integrate_array(element.axial_force, dx) / element.EA
        u1 = integrate_array(phi_neg1, dx)

        phi_neg2 = -integrate_array(element.axial_force[::-1], dx) / element.EA
        u2 = integrate_array(phi_neg2, dx)

        element.extension = -(u1 + u2) / 2.0
        element.max_extension = np.max(np.abs(element.extension))

        # assert element.N_1 is not None
        # assert element.N_2 is not None
        # u = 0.5 * (element.N_1 + element.N_2) / element.EA * element.l
        # du = u / con
        # element.extension = du * (np.arange(con) + 1)
