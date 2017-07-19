import copy
import math
import numpy as np

from StructuralEngineering.FEM.node import Node
from StructuralEngineering.basic import is_moving_towards


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

        for el in self.system.elements:
            # post processor element level
            self.post_el.node_results(el)

        count = 0
        for node in self.system.node_objects:
            for el in self.system.elements:
                # Minus sign, because the node force is opposite of the element force.
                if el.node_1.id == node.id:
                    self.system.node_objects[count] -= el.node_1
                elif el.node_2.id == node.id:
                    self.system.node_objects[count] -= el.node_2

            # Loads that are applied on the node of the support. Moment at a hinged support may not lead to reaction
            # moment
            for F_tuple in self.system.loads_moment:
                """
                tuple (nodeID, direction=3, Ty)
                """
                if F_tuple[0] == node.id:
                    self.system.node_objects[count].Ty += F_tuple[2]

                # The displacements are not summarized. Therefore the displacements are set for every node 1.
                # In order to ensure that every node is overwrote.
                if el.node_1.id == node.id:
                    self.system.node_objects[count].ux = el.node_1.ux
                    self.system.node_objects[count].uz = el.node_1.uz
                    self.system.node_objects[count].phi_y = el.node_1.phi_y
                if el.node_2.id == node.id:
                    self.system.node_objects[count].ux = el.node_2.ux
                    self.system.node_objects[count].uz = el.node_2.uz
                    self.system.node_objects[count].phi_y = el.node_2.phi_y
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

        for nodeID in supports:
            for node in self.system.node_objects:
                if nodeID == node.id:
                    node = copy.copy(node)
                    node.Fx *= -1
                    node.Fz *= -1
                    node.Ty *= -1
                    node.ux = None
                    node.uz = None
                    node.phi_y = None
                    self.system.reaction_forces.append(node)

    def element_results(self):
        """
        Determines the element results for al elements in the system on element level.
        """
        for el in self.system.elements:
            con = 100
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
        test_node = np.array([element.point_1.x, element.point_1.z])
        node_position = np.array([element.point_2.x, element.point_2.z])
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

            if element.q_load:
                q_part = (-0.5 * -element.q_load * x**2 + 0.5 * -element.q_load * element.l * x)
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

        deflection w =
        EI * (d^4/dx^4 * w(x)) = q

        solution of differential equation:

        w = 1/24 qx^4 + 1/6 c1x^3 + 1/2 c2 x^2 + c3x + c4  ---> c4 = w(0)
        phi = -1/6 qx^3/EI - 1/2c1x^2 -c2x -c3  ---> c3 = -phi
        M = EI(-1/2qx^2/EI -c1x -c2)  ---> c2 = -M/EI
        V = EI(-qx/EI -c1)  ---> c1 = -V/EI
        """

        if element.type == 'general':
            c1 = -element.shear_force[0] / element.EI
            c2 = -element.bending_moment[0] / element.EI
            c3 = element.node_1.phi_y
            c4 = 0
            w = np.empty(con)
            dx = element.l / con

            for i in range(con):
                x = (i + 1) * dx
                w[i] = 1 / 6 * c1 * x**3 + 0.5 * c2 * x**2 + c3 * x + c4
                if element.q_load:
                    w[i] += 1 / 24 * -element.q_load * x**4 / element.EI
            element.deflection = -w

            # max deflection
            element.max_deflection = max(abs(min(w)), abs(max(w)))

        """
        Extension
        """

        u = element.N / element.EA * element.l
        du = u / con

        element.extension = np.empty(con)

        for i in range(con):
            u = du * (i + 1)
            element.extension[i] = u



