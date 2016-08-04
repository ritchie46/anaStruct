import math
import copy
from StructuralEngineering.FEM.node import Node


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
        dx = element.point_1.x - element.point_2.x
        if math.isclose(dx, 0):  # element is vertical
            if element.point_1.z < element.point_2.z:  # point 1 is bottom
                element.N = -element.node_1.Fz  # compression and tension in opposite direction
            else:
                element.N = element.node_1.Fz  # compression and tension in opposite direction

        elif math.isclose(element.alpha, 0):  # element is horizontal
            if element.point_1.x < element.point_2.x:  # point 1 is left
                element.N = -element.node_1.Fx  # compression and tension in good direction
            else:
                element.N = element.node_1.Fx  # compression and tension in opposite direction

        else:
            if math.cos(element.alpha) > 0:
                if element.point_1.x < element.point_2.x:  # point 1 is left
                    if element.node_1.Fx > 0:  # compression in element
                        element.N = -math.sqrt(element.node_1.Fx ** 2 + element.node_1.Fz ** 2)
                    else:
                        element.N = math.sqrt(element.node_1.Fx ** 2 + element.node_1.Fz ** 2)
                else:  # point 1 is right
                    if element.node_1.Fx < 0:  # compression in element
                        element.N = -math.sqrt(element.node_1.Fx ** 2 + element.node_1.Fz ** 2)
                    else:
                        element.N = math.sqrt(element.node_1.Fx ** 2 + element.node_1.Fz ** 2)
