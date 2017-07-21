import math
import numpy as np
from StructuralEngineering.basic import converge, angle_x_axis
from StructuralEngineering.FEM.postprocess import SystemLevel as post_sl
from StructuralEngineering.FEM.elements import Element
from StructuralEngineering.FEM.node import Node
from StructuralEngineering.trigonometry import Pointxz, Pointxy
from StructuralEngineering.FEM.plotter import Plotter


class SystemElements:
    def __init__(self, figsize=(12, 8), xy_cs=True, EA=15e3, EI=5e3):
        # standard values if none provided
        self.EA = EA
        self.EI = EI
        self.node_ids = []
        self.max_node_id = 2  # minimum is 2, being 1 element
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
        self.figsize = figsize
        self.xy_cs = xy_cs
        # list of removed indexes due to conditions
        self.removed_indexes = []
        # list of indexes that remain after conditions are applied
        self.remainder_indexes = []
        # keep track of the node_id of the supports
        self.supports_fixed = []
        self.supports_hinged = []
        self.supports_roll = []
        self.supports_spring_x = []
        self.supports_spring_z = []
        self.supports_spring_y = []
        self.supports_roll_direction = []
        # keep track of the loads
        self.loads_point = []  # node ids with a point load
        self.loads_q = []  # element ids with a q-load
        self.loads_moment = []
        # results
        self.reaction_forces = {}  # node objects

        self.non_linear = False
        self.non_linear_elements = {}  # keys are element ids, values are dicts: {node_index: max moment capacity}
        self.element_map = {}
        self.node_map = {}
        self._spring_map = {}  # keys matrix index (for both row and columns), values K

    def add_truss_element(self, location_list, EA):
        return self.add_element(location_list, EA, 1e-14, type='truss')

    def add_element(self, location_list, EA=None, EI=None, hinge=None, mp=None, type="general"):
        """
        :param location_list: [[x, z], [x, z]] or [Pointxz, Pointxz]
        :param EA: (float) EA
        :param EI: (float) EI
        :param hinge: (integer) 1 or 2. Adds an hinge at the first or second node.
        :param mp: (dict) Set a maximum plastic moment capacity. Keys are integers representing the nodes. Values
                          are the bending moment capacity.
                          Example:
                          { 1: 210e3,
                            2: 180e3
                          }
        :return element id: (int)
        """
        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        # add the element number
        self.count += 1

        if isinstance(location_list[0], Pointxz):
            point_1 = location_list[0]
            point_2 = location_list[1]
        else:
            point_1 = Pointxz(location_list[0][0], location_list[0][1])
            point_2 = Pointxz(location_list[1][0], location_list[1][1])

        if self.xy_cs:
            point_1.z *= -1
            point_2.z *= -1

        node_id1 = 1
        node_id2 = 2
        existing_node1 = False
        existing_node2 = False

        if len(self.element_map) != 0:
            count = 1
            for el in self.elements:
                # check if node 1 of the element meets another node. If so, both have the same node_id
                if el.point_1 == point_1:
                    node_id1 = el.node_id1
                    existing_node1 = True
                elif el.point_2 == point_1:
                    node_id1 = el.node_id2
                    existing_node1 = True
                elif count == len(self.element_map) and existing_node1 is False:
                    self.max_node_id += 1
                    node_id1 = self.max_node_id

                # check for node 2
                if el.point_1 == point_2:
                    node_id2 = el.node_id1
                    existing_node2 = True
                elif el.point_2 == point_2:
                    node_id2 = el.node_id2
                    existing_node2 = True
                elif count == len(self.element_map) and existing_node2 is False:
                    self.max_node_id += 1
                    node_id2 = self.max_node_id
                count += 1

        # append the nodes to the system nodes list
        id1 = False
        id2 = False

        if node_id1 in self.node_map:
            id1 = True  # node_id1 already in system
        if node_id2 in self.node_map:
            id2 = True  # node_id1 already in system

        if id1 is False:
            node = Node(id=node_id1, point=point_1)
            self.node_ids.append(node_id1)
            self.node_map[node_id1] = node
        if id2 is False:
            node = Node(id=node_id2, point=point_2)
            self.node_ids.append(node_id2)
            self.node_map[node_id2] = node

        """
        Only one hinge per node. If not, the matrix cannot be solved.
        """

        if hinge == 1:
            if self.node_map[node_id1].hinge:
                hinge = None  # Ensure that there are not more than one hinge per node
            else:
                self.node_map[node_id1].hinge = True
        elif hinge == 2:
            if self.node_map[node_id2].hinge:
                hinge = None
            else:
                self.node_map[node_id2].hinge = True

        # determine the length of the elements
        point = point_2 - point_1
        l = point.modulus()

        # determine the angle of the element with the global x-axis
        delta_x = point_2.x - point_1.x
        delta_z = -point_2.z - -point_1.z  # minus sign to work with an opposite z-axis
        ai = angle_x_axis(delta_x, delta_z)

        # add element
        element = Element(self.count, EA, EI, l, ai, point_1, point_2, hinge)
        element.node_ids.append(node_id1)
        element.node_ids.append(node_id2)
        element.node_id1 = node_id1
        element.node_id2 = node_id2

        element.node_1 = Node(node_id1)
        element.node_2 = Node(node_id2)

        element.type = type

        self.elements.append(element)
        self.element_map[self.count] = element

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
        row_index_n1 = (element.node_1.id - 1) * 3

        # node 2
        row_index_n2 = (element.node_2.id - 1) * 3
        matrix_locations = []

        for _ in range(3):  # ux1, uz1, phi1
            full_row_locations = []
            column_index_n1 = (element.node_1.id - 1) * 3
            column_index_n2 = (element.node_2.id - 1) * 3

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
            column_index_n1 = (element.node_1.id - 1) * 3
            column_index_n2 = (element.node_2.id - 1) * 3

            for i in range(3):  # matrix row index 1, 2, 3
                full_row_locations.append((row_index_n2, column_index_n1))
                column_index_n1 += 1

            for i in range(3):  # matrix row index 4, 5, 6
                full_row_locations.append((row_index_n2, column_index_n2))
                column_index_n2 += 1

            row_index_n2 += 1
            matrix_locations.append(full_row_locations)
        self.system_matrix_locations.append(matrix_locations)

        if mp is not None:
            self.non_linear_elements[self.count] = mp
            self.non_linear = True

        return self.count

    def __assemble_system_matrix(self):
        """
        Shape of the matrix = n nodes * n d.o.f.
        Shape = n * 3
        """
        # if self.system_matrix is None:
        shape = len(self.node_ids) * 3
        self.shape_system_matrix = shape
        self.system_matrix = np.zeros((shape, shape))

        for matrix_index, K in self._spring_map.items():
            #  first index is row, second is column
            self.system_matrix[matrix_index][matrix_index] += K

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

    def solve(self, gnl=False, force_linear=False, verbosity=0):
        """

        :param gnl: (bool) Toggle geometrical non linear calculation.
        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0. get calculation outputs. 1. silence.
        :return: (array) Displacements vector.
        """

        if self.non_linear and not force_linear:
            return self._stiffness_adaptation(verbosity)

        if gnl:
            return self._gnl(verbosity)

        assert(self.system_force_vector is not None), "There are no forces on the structure"
        self.remainder_indexes = []
        self.removed_indexes = []
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
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

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
        assert(np.any(self.system_displacement_vector < 1e6)), "The displacements of the structure exceed 1e6. " \
                                                               "Check your support conditions," \
                                                               "or your elements Young's modulus"

        return self.system_displacement_vector

    def _stiffness_adaptation(self, verbosity):
        self.solve(False, True)
        if verbosity == 0:
            print("Starting stiffness adaptation calculation.")

        for c in range(1500):
            factors = []

            # update the elements stiffnesses
            for k, v in self.non_linear_elements.items():
                el = self.element_map[k]

                for node_no, mp in v.items():
                    m_e = el.element_force_vector[node_no * 3 - 1]

                    if abs(m_e) > mp:
                        el.nodes_plastic[node_no - 1] = True
                    if el.nodes_plastic[node_no - 1]:
                        factor = converge(m_e, mp, 3)
                        factors.append(factor)
                        el.update_stiffness(factor, node_no)

            if not np.allclose(factors, 1, 1e-3):
                self.solve(force_linear=True)
            else:
                break
        if verbosity == 0:
            print("Solved in {} iterations".format(c))

    def _gnl(self, verbosity):
        print(self.node_map[2])

        last_abs_u = np.abs(self.solve(False))
        if verbosity == 0:
            print("Starting geometrical non linear calculation")

        for c in range(10000):
            for el in self.element_map.values():
                delta_x = el.point_2.x + el.node_2.ux - el.point_1.x - el.node_1.ux
                # minus sign to work with an opposite z-axis
                delta_z = -el.point_2.z - el.node_2.uz + el.point_1.z + el.node_1.uz
                a_bar = angle_x_axis(delta_x, delta_z)
                ai = a_bar #+ el.node_1.phi_y
                aj = a_bar #+ el.node_2.phi_y
                l = math.sqrt(delta_x**2 + delta_z**2)
                el.compile_kinematic_matrix(ai, aj, l)
                #el.compile_constitutive_matrix(el.EA, el.EI, l)
                el.compile_stifness_matrix()

            current_abs_u = np.abs(self.solve(gnl=False))
            global_increase = np.sum(current_abs_u) / np.sum(last_abs_u)
            print(global_increase)
            if global_increase < 0.9:
                print(f"Divergence in {c} iterations")
                break
            if global_increase - 1 < 1e-9 and global_increase > 1:
                print(f"Convergence in {c} iterations")
                break


            last_abs_u = current_abs_u

    def _support_check(self, node_id):
        if self.node_map[node_id].hinge:
            raise Exception ("You cannot add a support to a hinged node.")

    def add_support_hinged(self, node_id):
        """
        adds a hinged support a the given node
        :param node_id: integer representing the nodes ID
        """
        self._support_check(node_id)
        self.set_displacement_vector([(node_id, 1), (node_id, 2)])

        # add the support to the support list for the plotter
        self.supports_hinged.append(self.node_map[node_id])

    def add_support_roll(self, node_id, direction=2):
        """
        adds a rollable support at the given node
        :param node_id: integer representing the nodes ID
        :param direction: integer representing the direction that is fixed: x = 1, z = 2
        """
        self._support_check(node_id)
        self.set_displacement_vector([(node_id, direction)])

        # add the support to the support list for the plotter
        self.supports_roll.append(self.node_map[node_id])
        self.supports_roll_direction.append(direction)

    def add_support_fixed(self, node_id):
        """
        adds a fixed support at the given node
        :param node_id: integer representing the nodes ID
        """
        self._support_check(node_id)
        self.set_displacement_vector([(node_id, 1), (node_id, 2), (node_id, 3)])

        # add the support to the support list for the plotter
        self.supports_fixed.append(self.node_map[node_id])

    def add_support_spring(self, node_id, translation, K):
        """
        :param translation: Integer representing prevented translation.
        1 = translation in x
        2 = translation in z
        3 = rotation in y
        :param node_id: Integer representing the nodes ID.
        :param K: Stiffness of the spring

        The stiffness of the spring is added in the system matrix at the location that represents the node and the
        displacement.
        """
        self._support_check(node_id)

        # determine the location in the system matrix
        # row and column are the same
        matrix_index = (node_id - 1) * 3 + translation - 1

        self._spring_map[matrix_index] = K

        # #  first index is row, second is column
        # self.system_matrix[matrix_index][matrix_index] += K

        if translation == 1:  # translation spring in x-axis
            self.set_displacement_vector([(node_id, 2)])

            # add the support to the support list for the plotter
            self.supports_spring_x.append(self.node_map[node_id])

        elif translation == 2:  # translation spring in z-axis
            self.set_displacement_vector([(node_id, 1)])

            # add the support to the support list for the plotter
            self.supports_spring_z.append(self.node_map[node_id])

        elif translation == 3:  # rotational spring in y-axis
            self.set_displacement_vector([(node_id, 1), (node_id, 2)])

            # add the support to the support list for the plotter
            self.supports_spring_y.append(self.node_map[node_id])

    def q_load(self, q, element_id, direction=1):
        """
        :param element_id: integer representing the element ID
        :param direction: 1 = towards the element, -1 = away from the element
        :param q: value of the q-load
        :return:
        """

        self.loads_q.append(element_id)
        element = self.element_map[element_id]
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
        self.set_force_vector([(element.node_1.id, 3, -left_moment), (element.node_2.id, 3, -right_moment),
                               (element.node_1.id, 2, reaction_z), (element.node_2.id, 2, reaction_z),
                               (element.node_1.id, 1, reaction_x), (element.node_2.id, 1, reaction_x)])

        return self.system_force_vector

    def point_load(self, Fx=0, Fz=0, node_id=None):
        self.loads_point.append((node_id, Fx, Fz))

        if node_id is not None:
            # system force vector.
            self.set_force_vector([(node_id, 1, Fx), (node_id, 2, Fz)])

        return self.system_force_vector

    def moment_load(self, Ty=0, node_id=None):
        self.loads_moment.append((node_id, 3, Ty))

        if node_id is not None:
            # system force vector.
            self.set_force_vector([(node_id, 3, Ty)])

    def show_structure(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True, supports=True):
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.plot_structure(figsize, verbosity, show, supports, scale, offset)

    def show_bending_moment(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.bending_moment(figsize, verbosity, scale, offset, show)

    def show_normal_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.normal_force(figsize, verbosity, scale, offset, show)

    def show_shear_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.shear_force(figsize, verbosity, scale, offset, show)

    def show_reaction_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                          linear=False, n_digits=2):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.displacements(factor, figsize, verbosity, scale, offset, show, linear)

    def get_node_results_system(self, node_id=0):
        """
        These are the node results. These are the opposite of the forces and displacements working on the elements and
        may seem counter intuitive.

        :param node_id: (integer) representing the node's ID. If integer = 0, the results of all nodes are returned
        :return:
                if node_id == 0: (list)
                    Returns a list containing tuples with the results
                    [(id, Fx, Fz, Ty, ux, uz, phi_y), (id, Fx, Fz...), () .. ]
                if node_id > 0: (dict)
        """
        result_list = []
        if node_id != 0:
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "Fx": node.Fx,
                "Fz": node.Fz,
                "Ty": node.Ty,
                "ux": node.ux,
                "uz": node.uz,
                "phi_y": node.phi_y
            }
        else:
            for node in self.node_map.values():
                result_list.append((node.id, node.Fx, node.Fz, node.Ty, node.ux, node.uz, node.phi_y))
        return result_list

    def get_node_displacements(self, node_id=0):
        """
        :param node_id: (integer) representing the node's ID. If integer = 0, the results of all nodes are returned
        :return:
                if node_id == 0: (list)
                    Returns a list containing tuples with the results
                    [(id,  ux, uz, phi_y), (id, ux, uz), () .. ]
                if node_id > 0: (dict)
        """
        if self.xy_cs:
            a = -1
        else:
            a = 1
        result_list = []
        if node_id != 0:
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "ux": -node.ux,
                "uz": a * node.uz,
                "phi_y": node.phi_y
            }
        else:
            for node in self.node_map.values():
                result_list.append((node.id, -node.ux, a * node.uz, node.phi_y))
        return result_list

    def get_element_results(self, element_id=0, verbose=False):
        """
        :param element_id: (int) representing the elements ID. If elementID = 0 the results of all elements are returned.
        :param verbose: (bool) If set to True the numerical results for the deflection and the bending moments are
                               returned.
        :return:
                if node_id == 0: (list)
                    Returns a list containing tuples with the results
                if node_id > 0: (dict)
        """
        if element_id != 0:
            el = self.element_map[element_id]
            if el.type == "truss":
                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.alpha,
                    "u": el.extension[0],
                    "N": el.N
                }
            else:
                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.alpha,
                    "u": el.extension[0],
                    "N": el.N,
                    "wmax": np.min(el.deflection),
                    "wmin": np.max(el.deflection),
                    "w": el.deflection if verbose else None,
                    "Mmin": np.min(el.bending_moment),
                    "Mmax": np.max(el.bending_moment),
                    "M": el.bending_moment if verbose else None,
                    "q": el.q_load
                }
        else:
            result_list = []
            for el in self.elements:

                if el.type == "truss":
                    result_list.append({
                        "id": el.id,
                        "length": el.l,
                        "alpha": el.alpha,
                        "u": el.extension[0],
                        "N": el.N
                    }
                    )

                else:
                    result_list.append(
                        {
                            "id": el.id,
                            "length": el.l,
                            "alpha": el.alpha,
                            "u": el.extension[0],
                            "N": el.N,
                            "wmax": np.min(el.deflection),
                            "wmin": np.max(el.deflection),
                            "w": el.deflection if verbose else None,
                            "Mmin": np.min(el.bending_moment),
                            "Mmax": np.max(el.bending_moment),
                            "M": el.bending_moment if verbose else None,
                            "q": el.q_load
                        }
                    )
            return result_list

