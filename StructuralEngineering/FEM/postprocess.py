import copy
import math
import numpy as np

from StructuralEngineering.FEM.node import Node
from StructuralEngineering.basic import is_moving_towards, integrate_array, angle_x_axis


class SystemLevel:
    def __init__(self, system):
        self.system = system
        # post processor element level
        self.post_el = ElementLevel(self.system)

    def node_results(self):
        """
        Determines the node results on the system level.
        Results placed in SystemElements class: self.node_objects (list).
        """

        for el in self.system.element_map.values():
            # post processor element level
            self.post_el.node_results(el)

        count = 0
        for node in self.system.node_map.values():
            # reset nodes in case of iterative calculation
            self.system.node_map[node.id].reset()

            for el in self.system.element_map.values():
                # Minus sign, because the node force is opposite of the element force.
                if el.node_1.id == node.id:
                    self.system.node_map[node.id] -= el.node_1
                    self.system.node_map[node.id].ux = -el.node_1.ux
                    self.system.node_map[node.id].uz = -el.node_1.uz
                    self.system.node_map[node.id].phi_y = -el.node_1.phi_y
                elif el.node_2.id == node.id:
                    self.system.node_map[node.id] -= el.node_2
                    self.system.node_map[node.id].ux = -el.node_2.ux
                    self.system.node_map[node.id].uz = -el.node_2.uz
                    self.system.node_map[node.id].phi_y = -el.node_2.phi_y

            # Loads that are applied on the node of the support. Moment at a hinged support may not lead to reaction
            # moment
            for F_tuple in self.system.loads_moment:
                """
                tuple (nodeID, direction=3, Ty)
                """
                if F_tuple[0] == node.id:
                    self.system.node_map[node.id].Ty += F_tuple[2]

                # The displacements are not summarized. Therefore the displacements are set for every node 1.
                # In order to ensure that every node is overwrote.
                if el.node_1.id == node.id:
                    self.system.node_map[node.id].ux = el.node_1.ux
                    self.system.node_map[node.id].uz = el.node_1.uz
                    self.system.node_map[node.id].phi_y = el.node_1.phi_y
                if el.node_2.id == node.id:
                    self.system.node_map[node.id].ux = el.node_2.ux
                    self.system.node_map[node.id].uz = el.node_2.uz
                    self.system.node_map[node.id].phi_y = el.node_2.phi_y
            count += 1

    def reaction_forces(self):
        supports = []
        for node in self.system.supports_fixed:
            supports.append(node.id)
        for node in self.system.supports_hinged:
            supports.append(node.id)
        for node in self.system.supports_roll:
            supports.append(node.id)
        for node in self.system.supports_spring_x:
            supports.append(node.id)
        for node in self.system.supports_spring_z:
            supports.append(node.id)
        for node in self.system.supports_spring_y:
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
            con = self.system.plotter.mesh
            self.post_el.determine_bending_moment(el, con)
            self.post_el.determine_shear_force(el, con)
            self.post_el.determine_displacements(el, con)


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self, element):
        """
        Determine node results on the element level.
        """

        element.node_1 = Node(
            id=element.node_ids[0],
            Fx=element.element_force_vector[0] + element.element_primary_force_vector[0],
            Fz=element.element_force_vector[1] + element.element_primary_force_vector[1],
            Ty=element.element_force_vector[2] + element.element_primary_force_vector[2],
            ux=element.element_displacement_vector[0],
            uz=element.element_displacement_vector[1],
            phi_y=element.element_displacement_vector[2],
        )

        element.node_2 = Node(
            id=element.node_ids[1],
            Fx=element.element_force_vector[3] + element.element_primary_force_vector[3],
            Fz=element.element_force_vector[4] + element.element_primary_force_vector[4],
            Ty=element.element_force_vector[5] + element.element_primary_force_vector[5],
            ux=element.element_displacement_vector[3],
            uz=element.element_displacement_vector[4],
            phi_y=element.element_displacement_vector[5]
        )
        self._determine_normal_force(element)

    @staticmethod
    def _determine_normal_force(element):
        test_node = np.array([element.vertex_1.x, element.vertex_1.z])
        node_position = np.array([element.vertex_2.x, element.vertex_2.z])
        displacement = np.array([element.node_2.ux - element.node_1.ux, element.node_2.uz - element.node_1.uz])

        force_towards = is_moving_towards(test_node, node_position, displacement)
        N = abs(math.sin(element.alpha) * element.node_1.Fz) + abs(math.cos(element.alpha) * element.node_1.Fx)

        if force_towards:
            element.N = -N
        else:
            element.N = N

    @staticmethod
    def determine_bending_moment(element, con):
        dT = -(element.node_2.Ty + element.node_1.Ty)  # T2 - (-T1)

        x_val = np.linspace(0, 1, con)
        m_val = np.empty(con)

        # determine moment for 0 < x < length of the element
        count = 0
        for i in x_val:
            x = i * element.l
            x_val[count] = x
            m_val[count] = element.node_1.Ty + i * dT

            if element.all_q_load:
                q = element.all_q_load
                q_part = (-0.5 * -q * x**2 + 0.5 * -q * element.l * x)
                m_val[count] += q_part
            count += 1
        element.bending_moment = m_val

    @staticmethod
    def determine_shear_force(element, con):
        """
        Determines the shear force by differentiating the bending moment.
        :param element: (object) of the Element class
        """
        dV = np.diff(element.bending_moment)
        dx = element.l / (con - 1)
        shear_force = dV / dx

        # Due to differentiation the first and the last values must be corrected.
        correction = shear_force[1] - shear_force[0]
        shear_force = np.insert(shear_force, 0, [shear_force[0] - 0.5 * correction])
        shear_force = np.insert(shear_force, con, [shear_force[-1] + 0.5 * correction])
        element.shear_force = shear_force

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

        if element.type == 'general':
            dx = element.l / (len(element.bending_moment) - 1)
            phi_neg = -integrate_array(element.bending_moment / element.EI, dx)
            w = integrate_array(phi_neg, dx)

            # Angle between last w and elements axis. The w array will be corrected so that this angle == 0.
            alpha = math.atan(w[-1] / element.l)

            lx = np.linspace(0, element.l, con)

            w = w - lx * np.sin(alpha)
            element.deflection = -w
            element.max_deflection = np.max(np.abs(w))

        """
        Extension
        """

        u = element.N / element.EA * element.l
        du = u / con

        element.extension = np.empty(con)

        for i in range(con):
            u = du * (i + 1)
            element.extension[i] = u



