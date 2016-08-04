import copy
import math

import numpy as np

from StructuralEngineering.FEM.node import Node
from StructuralEngineering.basic import is_moving_towards


class SystemLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
        """
        Determines the node results on the system level.
        Results placed in SystemElements class: self.node_objects (list).
        """

        # post processor element level
        post_el = ElementLevel(self.system)
        post_el.node_results()

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


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
        """
        Determine node results on the element level.
        """
        for el in self.system.elements:
            el.node_1 = Node(
                ID=el.node_ids[0],
                Fx=el.element_force_vector[0] + el.element_primary_force_vector[0],
                Fz=el.element_force_vector[1] + el.element_primary_force_vector[1],
                Ty=el.element_force_vector[2] + el.element_primary_force_vector[2],
                ux=el.element_displacement_vector[0],
                uz=el.element_displacement_vector[1],
                phi_y=el.element_displacement_vector[2],
            )

            el.node_2 = Node(
                ID=el.node_ids[1],
                Fx=el.element_force_vector[3] + el.element_primary_force_vector[3],
                Fz=el.element_force_vector[4] + el.element_primary_force_vector[4],
                Ty=el.element_force_vector[5] + el.element_primary_force_vector[5],
                ux=el.element_displacement_vector[3],
                uz=el.element_displacement_vector[4],
                phi_y=el.element_displacement_vector[5]
            )
            self._determine_normal_force(el)

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
