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
                if el.node_1.ID == node.ID:
                    self.system.node_objects[count] -= el.node_1
                elif el.node_2.ID == node.ID:
                    self.system.node_objects[count] -= el.node_2

            # The displacements are the same direction, therefore the -1 multiplication
            self.system.node_objects[count].ux *= -1
            self.system.node_objects[count].uz *= -1
            self.system.node_objects[count].phi_y *= -1
            count += 1

    def reaction_forces(self):
        supports = []
        for node in self.system.supports_fixed:
            supports.append(node.ID)
        for node in self.system.supports_hinged:
            supports.append(node.ID)
        for node in self.system.supports_roll:
            supports.append(node.ID)
        for node in self.system.supports_spring_x:
            supports.append(node.ID)
        for node in self.system.supports_spring_z:
            supports.append(node.ID)
        for node in self.system.supports_spring_y:
            supports.append(node.ID)

        for nodeID in supports:
            for node in self.system.node_objects:
                if nodeID == node.ID:
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
            self.post_el.determine_bending_moment(el, 10)
            self.post_el.determine_shear_force(el)


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self, element):
        """
        Determine node results on the element level.
        """
        element.node_1 = Node(
            ID=element.node_ids[0],
            Fx=element.element_force_vector[0] + element.element_primary_force_vector[0],
            Fz=element.element_force_vector[1] + element.element_primary_force_vector[1],
            Ty=element.element_force_vector[2] + element.element_primary_force_vector[2],
            ux=element.element_displacement_vector[0],
            uz=element.element_displacement_vector[1],
            phi_y=element.element_displacement_vector[2],
        )

        element.node_2 = Node(
            ID=element.node_ids[1],
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
    def determine_shear_force(element):
        """
        Determines the shear force by differentiating the bending moment
        :param element: (object) of the Element class
        """
        dV = np.diff(element.bending_moment)
        length = len(element.bending_moment)
        dx = element.l / (length - 1)
        shear_force = dV / dx

        # Due to differentiation the first and the last values must be corrected.
        correction = shear_force[1] - shear_force[0]
        shear_force = np.insert(shear_force, 0, [shear_force[0] - 0.5 * correction])
        shear_force = np.insert(shear_force, length, [shear_force[-1] + 0.5 * correction])
        element.shear_force = shear_force
