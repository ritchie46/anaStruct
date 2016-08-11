import math
import numpy as np
from StructuralEngineering.FEM.postprocess import SystemLevel as post_sl
from StructuralEngineering.FEM.elements import Element
from StructuralEngineering.FEM.node import Node
from StructuralEngineering.trigonometry import Point
from StructuralEngineering.FEM.plotter import Plotter


class SystemElements:
    def __init__(self):
        self.node_ids = []
        self.max_node_id = 2  # minimum is 2, being 1 element
        self.node_objects = []
        self.elements = []
        self.count = 0
        self.system_matrix_locations = []
        self.system_matrix = None
        self.system_force_vector = None
        self.system_displacement_vector = None
        self.shape_system_matrix = None
        self.reduced_force_vector = None
        self.reduced_system_matrix = None
        self.post_processor = post_sl(self)
        self.plotter = Plotter(self)
        # list of removed indexes due to conditions
        self.removed_indexes = []
        # list of indexes that remain after conditions are applied
        self.remainder_indexes = []
        # keep track of the nodeID of the supports
        self.supports_fixed = []
        self.supports_hinged = []
        self.supports_roll = []
        self.supports_spring_x = []
        self.supports_spring_z = []
        self.supports_spring_y = []
        # keep track of the loads
        self.loads_point = []  # node ids with a point load
        self.loads_q = []  # element ids with a q-load
        self.loads_moment = []
        # results
        self.reaction_forces = []  # node objects

    def add_truss_element(self, location_list, EA):
        self.add_element(location_list, EA, 1e-14, type='truss')

    def add_element(self, location_list, EA, EI, **kwargs):
        """
        :param location_list: [[x, z], [x, z]]
        :param EA: EA
        :param EI: EI
        :return: Void
        """
        # add the element number
        self.count += 1
        point_1 = Point(location_list[0][0], location_list[0][1])
        point_2 = Point(location_list[1][0], location_list[1][1])

        nodeID1 = 1
        nodeID2 = 2
        existing_node1 = False
        existing_node2 = False

        if len(self.elements) != 0:
            count = 1
            for el in self.elements:
                # check if node 1 of the element meets another node. If so, both have the same nodeID
                if el.point_1 == point_1:
                    nodeID1 = el.nodeID1
                    existing_node1 = True
                elif el.point_2 == point_1:
                    nodeID1 = el.nodeID2
                    existing_node1 = True
                elif count == len(self.elements) and existing_node1 is False:
                    self.max_node_id += 1
                    nodeID1 = self.max_node_id

                # check for node 2
                if el.point_1 == point_2:
                    nodeID2 = el.nodeID1
                    existing_node2 = True
                elif el.point_2 == point_2:
                    nodeID2 = el.nodeID2
                    existing_node2 = True
                elif count == len(self.elements) and existing_node2 is False:
                    self.max_node_id += 1
                    nodeID2 = self.max_node_id
                count += 1

        # append the nodes to the system nodes list
        id1 = False
        id2 = False
        for i in self.node_ids:
            if i == nodeID1:
                id1 = True  # nodeID1 allready in system
            if i == nodeID2:
                id2 = True  # nodeID2 allready in system

        if id1 is False:
            self.node_ids.append(nodeID1)
            self.node_objects.append(Node(ID=nodeID1, point=point_1))
        if id2 is False:
            self.node_ids.append(nodeID2)
        self.node_objects.append(Node(ID=nodeID2, point=point_2))

        # determine the length of the elements
        point = point_2 - point_1
        l = point.modulus()

        # determine the angle of the element with the global x-axis
        delta_x = point_2.x - point_1.x
        delta_z = -point_2.z - -point_1.z  # minus sign to work with an opposite z-axis

        if math.isclose(delta_x, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is vertical
            if delta_z < 0:
                ai = 1.5 * math.pi
            else:
                ai = 0.5 * math.pi
        elif math.isclose(delta_z, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is horizontal
            if delta_x > 0:
                ai = 0
            else:
                ai = math.pi
        elif delta_x > 0 and delta_z > 0:  # quadrant 1 of unity circle
            ai = math.atan(abs(delta_z) / abs(delta_x))
        elif delta_x < 0 < delta_z:  # quadrant 2 of unity circle
            ai = 0.5 * math.pi + math.atan(abs(delta_x) / abs(delta_z))
        elif delta_x < 0 and delta_z < 0:  # quadrant 3 of unity circle
            ai = math.pi + math.atan(abs(delta_z) / abs(delta_x))
        elif delta_z < 0 < delta_x:  # quadrant 4 of unity circle
            ai = 1.5 + math.pi + math.atan(abs(delta_x) / abs(delta_z))
        else:
            raise ValueError("Can't determine the angle of the given element")

        # aj = ai
        # add element
        element = Element(self.count, EA, EI, l, ai, ai, point_1, point_2)
        element.node_ids.append(nodeID1)
        element.node_ids.append(nodeID2)
        element.nodeID1 = nodeID1
        element.nodeID2 = nodeID2

        element.node_1 = Node(nodeID1)
        element.node_2 = Node(nodeID2)

        element.type = kwargs.get('type', 'general')  # searches key 'type', otherwise set default value 'general'

        self.elements.append(element)

        """
        system matrix [K]

        [fx 1] [K         |  \ node 1 starts at row 1
        |fz 1] |  K       |  /
        |Ty 1] |   K      | /
        |fx 2] |    K     |  \ node 2 starts at row 4
        |fz 2] |     K    |  /
        |Ty 2] |      K   | /
        |fx 3] |       K  |  \ node 3 starts at row 7
        |fz 3] |        K |  /
        [Ty 3] [         K] /

                n   n  n
                o   o  o
                d   d  d
                e   e  e
                1   2  3

        thus with appending numbers in the system matrix: column = row
        """
        # starting row
        # node 1
        row_index_n1 = (element.node_1.ID - 1) * 3

        # node 2
        row_index_n2 = (element.node_2.ID - 1) * 3
        matrix_locations = []

        for _ in range(3):  # ux1, uz1, phi1
            full_row_locations = []
            column_index_n1 = (element.node_1.ID - 1) * 3
            column_index_n2 = (element.node_2.ID - 1) * 3

            for i in range(3):  # matrix row index 1, 2, 3
                full_row_locations.append((row_index_n1, column_index_n1))
                column_index_n1 += 1

            for i in range(3):  # matrix row index 4, 5, 6
                full_row_locations.append((row_index_n1, column_index_n2))
                column_index_n2 += 1

            row_index_n1 += 1
            matrix_locations.append(full_row_locations)

        for _ in range(3):  # ux3, uz3, phi3
            full_row_locations = []
            column_index_n1 = (element.node_1.ID - 1) * 3
            column_index_n2 = (element.node_2.ID - 1) * 3

            for i in range(3):  # matrix row index 1, 2, 3
                full_row_locations.append((row_index_n2, column_index_n1))
                column_index_n1 += 1

            for i in range(3):  # matrix row index 4, 5, 6
                full_row_locations.append((row_index_n2, column_index_n2))
                column_index_n2 += 1

            row_index_n2 += 1
            matrix_locations.append(full_row_locations)
        self.system_matrix_locations.append(matrix_locations)

    def __assemble_system_matrix(self):
        """
        Shape of the matrix = n nodes * n d.o.f.
        Shape = n * 3
        :return:
        """
        if self.system_matrix is None:
            shape = len(self.node_ids) * 3
            self.shape_system_matrix = shape
            self.system_matrix = np.zeros((shape, shape))

        for i in range(len(self.elements)):

            for row_index in range(len(self.system_matrix_locations[i])):
                count = 0
                for loc in self.system_matrix_locations[i][row_index]:
                    self.system_matrix[loc[0]][loc[1]] += self.elements[i].stiffness_matrix[row_index][count]
                    count += 1

        # returns True if symmetrical.
        return np.allclose((self.system_matrix.transpose()), self.system_matrix)

    def set_force_vector(self, force_list):
        """
        :param force_list: list containing tuples with the
        1. number of the node,
        2. the number of the direction (1 = x, 2 = z, 3 = y)
        3. the force
        [(1, 3, 1000)] node=1, direction=3 (y), force=1000
        list may contain multiple tuples
        :return: Vector with forces on the nodes
        """
        if self.system_force_vector is None:
            self.system_force_vector = np.zeros(self.max_node_id * 3)

        for i in force_list:
            # index = number of the node-1 * nd.o.f. + x, or z, or y
            # (x = 1 * i[1], y = 2 * i[1]
            index = (i[0] - 1) * 3 + i[1] - 1
            # force = i[2]
            self.system_force_vector[index] += i[2]
        return self.system_force_vector

    def set_displacement_vector(self, nodes_list):
        """
        :param nodes_list: list containing tuples with
        1.the node
        2. the d.o.f. that is set
        :return: Vector with the displacements of the nodes (If displacement is not known, the value is set
        to NaN)
        """
        if self.system_displacement_vector is None:
            self.system_displacement_vector = np.empty(self.max_node_id * 3)
            self.system_displacement_vector[:] = np.NaN

        for i in nodes_list:
            index = (i[0] - 1) * 3 + i[1] - 1
            self.system_displacement_vector[index] = 0
        return self.system_displacement_vector

    def __process_conditions(self):
        original_force_vector = np.array(self.system_force_vector)
        original_system_matrix = np.array(self.system_matrix)

        remove_count = 0
        # remove the unsolvable values from the matrix and vectors
        for i in range(self.shape_system_matrix):
            index = i - remove_count
            if self.system_displacement_vector[index] == 0:
                self.system_displacement_vector = np.delete(self.system_displacement_vector, index, 0)
                self.system_force_vector = np.delete(self.system_force_vector, index, 0)
                self.system_matrix = np.delete(self.system_matrix, index, 0)
                self.system_matrix = np.delete(self.system_matrix, index, 1)
                remove_count += 1
                self.removed_indexes.append(i)
            else:
                self.remainder_indexes.append(i)
        self.reduced_force_vector = self.system_force_vector
        self.reduced_system_matrix = self.system_matrix
        self.system_force_vector = original_force_vector
        self.system_matrix = original_system_matrix

    def solve(self):
        assert(self.system_force_vector is not None), "There are no forces on the structure"
        self.__assemble_system_matrix()
        self.__process_conditions()

        # solution of the reduced system (reduced due to support conditions)
        reduced_displacement_vector = np.linalg.solve(self.reduced_system_matrix, self.reduced_force_vector)

        # add the solution of the reduced system in the complete system displacement vector
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        count = 0
        for i in self.remainder_indexes:
            self.system_displacement_vector[i] = reduced_displacement_vector[count]
            count += 1

        # determine the displacement vector of the elements
        for el in self.elements:
            index_node_1 = (el.node_1.ID - 1) * 3
            index_node_2 = (el.node_2.ID - 1) * 3

            for i in range(3):  # node 1 ux, uz, phi
                el.element_displacement_vector[i] = self.system_displacement_vector[index_node_1 + i]

            for i in range(3):  # node 2 ux, uz, phi
                el.element_displacement_vector[3 + i] = self.system_displacement_vector[index_node_2 + i]

            el.determine_force_vector()

        # determining the node results in post processing class
        self.post_processor.node_results()
        self.post_processor.reaction_forces()
        self.post_processor.element_results()

        # check the values in the displacement vector for extreme values, indicating a flawed calculation
        for value in np.nditer(self.system_displacement_vector):
            assert(value < 1e6), "The displacements of the structure exceed 1e6. Check your support conditions," \
                                 "or your elements Young's modulus"
        return self.system_displacement_vector

    def add_support_hinged(self, nodeID):
        """
        adds a hinged support a the given node
        :param nodeID: integer representing the nodes ID
        """
        self.set_displacement_vector([(nodeID, 1), (nodeID, 2)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == nodeID:
                self.supports_hinged.append(obj)
                break

    def add_support_roll(self, nodeID, direction=2):
        """
        adds a rollable support at the given node
        :param nodeID: integer representing the nodes ID
        :param direction: integer representing the direction that is fixed: x = 1, z = 2
        """
        self.set_displacement_vector([(nodeID, direction)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == nodeID:
                self.supports_roll.append(obj)
                break

    def add_support_fixed(self, nodeID):
        """
        adds a fixed support at the given node
        :param nodeID: integer representing the nodes ID
        """
        self.set_displacement_vector([(nodeID, 1), (nodeID, 2), (nodeID, 3)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == nodeID:
                self.supports_fixed.append(obj)
                break

    def add_support_spring(self, nodeID, translation, K):
        """
        :param translation: Integer representing prevented translation.
        1 = translation in x
        2 = translation in z
        3 = rotation in y
        :param nodeID: Integer representing the nodes ID.
        :param K: Stiffness of the spring

        The stiffness of the spring is added in the system matrix at the location that represents the node and the
        displacement.
        """
        if self.system_matrix is None:
            shape = len(self.node_ids) * 3
            self.shape_system_matrix = shape
            self.system_matrix = np.zeros((shape, shape))

        # determine the location in the system matrix
        # row and column are the same
        matrix_index = (nodeID - 1) * 3 + translation - 1

        #  first index is row, second is column
        self.system_matrix[matrix_index][matrix_index] += K

        if translation == 1:  # translation spring in x-axis
            self.set_displacement_vector([(nodeID, 2)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == nodeID:
                    self.supports_spring_x.append(obj)
                    break
        elif translation == 2:  # translation spring in z-axis
            self.set_displacement_vector([(nodeID, 1)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == nodeID:
                    self.supports_spring_z.append(obj)
                    break
        elif translation == 3:  # rotational spring in y-axis
            self.set_displacement_vector([(nodeID, 1), (nodeID, 2)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == nodeID:
                    self.supports_spring_y.append(obj)
                    break

    def q_load(self, q, elementID, direction=1):
        """
        :param elementID: integer representing the element ID
        :param direction: 1 = towards the element, -1 = away from the element
        :param q: value of the q-load
        :return:
        """

        self.loads_q.append(elementID)

        for obj in self.elements:
            if obj.ID == elementID:
                element = obj
                break

        element.q_load = q * direction

        # determine the left point for the direction of the primary moment
        left_moment = 1 / 12 * q * element.l**2 * direction
        right_moment = -1 / 12 * q * element.l**2 * direction
        reaction_x = 0.5 * q * element.l * direction * math.sin(element.alpha)
        reaction_z = 0.5 * q * element.l * direction * math.cos(element.alpha)

        # place the primary forces in the 6*6 primary force matrix
        element.element_primary_force_vector[2] += left_moment
        element.element_primary_force_vector[5] += right_moment

        element.element_primary_force_vector[1] -= reaction_z
        element.element_primary_force_vector[4] -= reaction_z
        element.element_primary_force_vector[0] -= reaction_x
        element.element_primary_force_vector[3] -= reaction_x

        # system force vector. System forces = element force * -1
        # first row are the moments
        # second and third row are the reacton forces. The direction
        self.set_force_vector([(element.node_1.ID, 3, -left_moment), (element.node_2.ID, 3, -right_moment),
                               (element.node_1.ID, 2, reaction_z), (element.node_2.ID, 2, reaction_z),
                               (element.node_1.ID, 1, reaction_x), (element.node_2.ID, 1, reaction_x)])

        return self.system_force_vector

    def point_load(self, Fx=0, Fz=0, nodeID=None):
        self.loads_point.append((nodeID, Fx, Fz))

        if nodeID is not None:
            # system force vector.
            self.set_force_vector([(nodeID, 1, Fx), (nodeID, 2, Fz)])

        return self.system_force_vector

    def moment_load(self, Ty=0, nodeID=None):
        self.loads_moment.append((nodeID, 3, Ty))

        if nodeID is not None:
            # system force vector.
            self.set_force_vector([(nodeID, 3, Ty)])

    def show_structure(self):
        self.plotter.plot_structure(plot_now=True)

    def show_bending_moment(self):
        self.plotter.bending_moment()

    def show_normal_force(self):
        self.plotter.normal_force()

    def show_shear_force(self):
        self.plotter.shear_force()

    def show_reaction_force(self):
        self.plotter.reaction_force()

    def show_displacement(self):
        self.plotter.displacements()

    def get_node_results_system(self, nodeID=0):
        """
        :param nodeID: (integer) representing the node's ID. If integer = 0, the results of all nodes are returned
        :return:
                if nodeID == 0: (list)
                    Returns a list containing tuples with the results
                if nodeID > 0: (tuple)
                    indexes:
                    0: node's ID
                    1: Fx
                    2: Fz
                    3: Ty
                    4: ux
                    5: uz
                    6: phi_y
        """
        result_list = []
        for obj in self.node_objects:
            if obj.ID == nodeID:
                return obj.ID, obj.Fx, obj.Fz, obj.Ty, obj.ux, obj.uz, obj.phi_y
            else:
                result_list.append((obj.ID, obj.Fx, obj.Fz, obj.Ty, obj.ux, obj.uz, obj.phi_y))
        return result_list

    def get_element_results(self, elementID=0):
        """
        :param elementID: (int) representing the elements ID. If elementID = 0 the results of all elements are returned.
        :return:
                if nodeID == 0: (list)
                    Returns a list containing tuples with the results
                if nodeID > 0: (tuple)
                    indexes:
                    0: elements ID
                    1: length of the element
                    2: elements angle with the global x-axis is radians
                    3: extension
                    4: normal force
                    5: absolute value of maximum deflection
                    6: absolute value of maximum bending moment
                    7: absolute value of maximum shear force
                    8: q-load applying on the element
        """
        result_list = []
        for el in self.elements:
            if elementID == el.ID:
                if el.type == "truss":
                    return (el.ID,
                        el.l,
                        el.alpha,
                        el.extension[0],
                        el.N,
                        None,
                        None,
                        None,
                        None
                            )
                else:
                    return (el.ID,
                            el.l,
                            el.alpha,
                            el.extension[0],
                            el.N,
                            max(abs(min(el.deflection)), abs(max(el.deflection))),
                            max(abs(min(el.bending_moment)), abs(max(el.bending_moment))),
                            max(abs(min(el.shear_force)), abs(max(el.shear_force))),
                            el.q_load
                            )
            else:
                if el.type == "truss":
                    result_list.append(
                        (el.ID,
                         el.l,
                         el.alpha,
                         el.extension[0],
                         el.N,
                         None,
                         None,
                         None,
                         None
                         )
                    )

                else:
                    result_list.append((
                        el.ID,
                        el.l,
                        el.alpha,
                        el.extension[0],
                        el.N,
                        max(abs(min(el.deflection)), abs(max(el.deflection))),
                        max(abs(min(el.bending_moment)), abs(max(el.bending_moment))),
                        max(abs(min(el.shear_force)), abs(max(el.shear_force))),
                        el.q_load
                    )
                    )
        return result_list
