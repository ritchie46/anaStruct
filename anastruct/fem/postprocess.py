import copy
import math
import numpy as np
import functions

from anastruct.fem.node import Node
from anastruct.basic import integrate_array


class SystemLevel:
    def __init__(self, system):
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

            for i in range(len(v)):
                node = v[i].node_map[k]
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
        Determines the element results for al elements in the system on element level.
        """

        for el in self.system.element_map.values():
            self.post_el.determine_axial_force(el)
            con = self.system.plotter.mesh
            self.post_el.determine_bending_moment(el, con)
            self.post_el.determine_shear_force(el, con)
            self.post_el.determine_displacements(el, con)


class ElementLevel:
    def __init__(self, system):
        self.system = system

    @staticmethod
    def node_results(element):
        """
        Determine node results on the element level.

        """
        # Global coordinates system
        element.node_map[element.node_id1] = Node(
            id=element.node_id1,
            Fx=element.element_force_vector[0]
            + element.element_primary_force_vector[0],
            Fz=element.element_force_vector[1]
            + element.element_primary_force_vector[1],
            Ty=element.element_force_vector[2]
            + element.element_primary_force_vector[2],
            ux=element.element_displacement_vector[0],
            uz=element.element_displacement_vector[1],
            phi_y=element.element_displacement_vector[2],
        )
        element.node_map[element.node_id2] = Node(
            id=element.node_id2,
            Fx=element.element_force_vector[3]
            + element.element_primary_force_vector[3],
            Fz=element.element_force_vector[4]
            + element.element_primary_force_vector[4],
            Ty=element.element_force_vector[5]
            + element.element_primary_force_vector[5],
            ux=element.element_displacement_vector[3],
            uz=element.element_displacement_vector[4],
            phi_y=element.element_displacement_vector[5],
        )

        # Local coordinate system. With inclined supports
        for i in range(1, 3):
            a_n = getattr(element, "a{}".format(i))
            if a_n != element.angle:
                node = element.node_map[getattr(element, "node_id{}".format(i))]
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
    def determine_axial_force(element):
        N_1 = (math.sin(element.angle) * element.node_1.Fz) + -(
            math.cos(element.angle) * element.node_1.Fx
        )
        N_2 = -(math.sin(element.angle) * element.node_2.Fz) + (
            math.cos(element.angle) * element.node_2.Fx
        )

        element.N_1 = N_1
        element.N_2 = N_2

    @staticmethod
    def determine_bending_moment(element, con):
        dT = -(element.node_2.Ty + element.node_1.Ty)  # T2 - (-T1)

        iteration_factor = np.linspace(0, 1, con)
        x = iteration_factor * element.l
        m_val = element.node_1.Ty + iteration_factor * dT
        if element.all_q_load:
            q = element.all_q_load
            qi = element.all_qi_load
            if q == qi:
                q_part = -0.5 * -q * x ** 2 + 0.5 * -q * element.l * x
            elif q != qi:
                if q == 0 or qi == 0:
                    q_part = q / (6 * element.l) * (x - element.vertex_1.x) ** 3 - q * 0.5 * element.l * 1/3 * x
                elif q != 0 and qi != 0:
                    q_part = 0.5 * qi * (x - element.vertex_1.x) ** 2 + (q - qi) / (6 * element.l) * \
                             (x - element.vertex_1.x) ** 3 + -(((q + qi) / 2) + qi / 2) * element.l * (1 / 3) * x
            m_val += q_part

        functions.eq.append(np.polyfit(x, m_val, 3))

        element.bending_moment = m_val

    @staticmethod
    def determine_shear_force(element, con):
        """
        Determines the shear force by differentiating the bending moment.
        :param element: (object) of the Element class
        """
        # dV = np.diff(element.bending_moment)
        # dx = element.l / (con - 1)
        # shear_force = dV / dx
        #
        # # Due to differentiation the first and the last values must be corrected.
        # correction = shear_force[1] - shear_force[0]
        # shear_force = np.insert(shear_force, 0, [shear_force[0] - 0.5 * correction])
        # shear_force = np.insert(shear_force, con, [shear_force[-1] + 0.5 * correction])
        # element.shear_force = shear_force

        iteration_factor = np.linspace(0, 1, con)
        x = iteration_factor * element.l
        eq = functions.eq
        c = functions.c
        shear_force = eq[c][0] * 3 * x ** 2 + eq[c][1] * 2 * x + eq[c][2]
        element.shear_force = shear_force
        c += 1


    @staticmethod
    def determine_displacements(element, con):
        """
        Determines the displacement by integrating the bending moment.
        :param element: (object) of the Element class

        w = -M''

        This gives you the formula

        w = -aMx +bx + c

        a = already defined by the integral
        b = Scale the slope of the parabola. This is the rotation of the deflection. You can think of this as
            the angle of the deflection beam. By rotating the beam so that the last deflection w = 0 you get the correct
            value for b. w[-1] = 0.
        c = Translate the parabola. Translate it so that w[0] = 0
        """

        if element.type == "general":
            dx = element.l / (len(element.bending_moment) - 1)

            # Take the average of cumulative integration from both sides. This is due to numerical differences, that
            # would be observable in symmetrical structures.
            phi_neg = (
                -0.5
                * (
                    integrate_array(element.bending_moment, dx)
                    + integrate_array(element.bending_moment[::-1], dx)
                )
                / element.EI
            )
            w = integrate_array(phi_neg, dx)

            # Angle between last w and elements axis. The w array will be corrected so that this angle == 0.
            alpha = np.arctan(w[-1] / element.l)

            lx = np.linspace(0, element.l, con)

            w = w - lx * np.tan(alpha)
            element.deflection = -w
            element.max_deflection = np.max(np.abs(w))

        # Extension
        u = 0.5 * (element.N_1 + element.N_2) / element.EA * element.l
        du = u / con
        element.extension = du * (np.arange(con) + 1)
