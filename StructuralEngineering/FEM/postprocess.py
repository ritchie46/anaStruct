import math
from StructuralEngineering.FEM.node import Node


class SystemLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
        post_el = ElementLevel(self.system)
        post_el.node_results()

        count = 0
        for node in self.system.node_objects:
            for el in self.system.elements:
                if el.node_1.ID == node.ID:
                    self.system.node_objects[count] += el.node_1
                elif el.node_2.ID == node.ID:
                    self.system.node_objects[count] += el.node_2
            count += 1


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
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