import math
import numpy as np
from StructuralEngineering.basic import converge, angle_x_axis, FEMException
from StructuralEngineering.FEM.postprocess import SystemLevel as post_sl
from StructuralEngineering.FEM.elements import Element, det_moment, det_shear
from StructuralEngineering.FEM.node import Node
from StructuralEngineering.vertex import Vertex_xz
from StructuralEngineering.FEM.plotter import Plotter


class SystemElements:
    def __init__(self, figsize=(12, 8), xy_cs=True, EA=15e3, EI=5e3, load_factor=1, mesh=50, plot_backend='mpl'):
        """

        :param figsize: (tpl)
        :param xy_cs: (Bool) Convert to xy coordinate system. Standard is xz
        :param EA: (flt) Standard E * A
        :param EI: (flt) Standard E * I
        :param load_factor: (tpl) Multiply all loads with this factor.
        :param mesh: (int) Plotting mesh. Has no influence on the calculation.
        :param plot_backend: (str)  matplotlib  -> "mpl"
                                    plotly      -> "plt"
                                    plotly nb   -> "ipt"
        """
        # init object
        self.post_processor = post_sl(self)
        self.plotter = Plotter(self, mesh, plot_backend)

        # standard values if none provided
        self.EA = EA
        self.EI = EI
        self.figsize = figsize
        self.xy_cs = xy_cs

        # structure system
        self.element_map = {}  # maps element ids to the Element objects.
        self.node_map = {}  # maps node ids to the Node objects.
        self.system_spring_map = {}  # keys matrix index (for both row and columns), values K

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
        self.loads_q = {}  # element ids with a q-load
        self.loads_moment = []
        self.loads_dead_load = []  # element ids with q-load due to dead load

        # results
        self.reaction_forces = {}  # node objects
        self.non_linear = False
        self.non_linear_elements = {}  # keys are element ids, values are dicts: {node_index: max moment capacity}

        # previous point of element
        self._previous_point = Vertex_xz(0, 0)
        self.load_factor = load_factor

        # Objects state
        self.max_node_id = 2  # minimum is 2, being 1 element
        self.count = 0
        self.system_matrix_locations = []
        self.system_matrix = None
        self.system_force_vector = None
        self.system_displacement_vector = None
        self.shape_system_matrix = None
        self.reduced_force_vector = None
        self.reduced_system_matrix = None

    def add_truss_element(self, location_list, EA=None):
        return self.add_element(location_list, EA, 1e-14, element_type='truss')

    def add_element(self, location_list, EA=None, EI=None, g=0, mp=None, spring=None, **kwargs):
        """
        :param location_list: (list/ Pointxz) The two nodes of the element or the next node of the element.
                                Examples:
                                    [[x, z], [x, z]]
                                    [Pointxz, Pointxz]
                                    [x, z]
                                    Pointxz
        :param EA: (flt) EA
        :param EI: (flt) EI
        :param g: (flt) Weight per meter. [kN/m] / [N/m]
        :param mp: (dict) Set a maximum plastic moment capacity. Keys are integers representing the nodes. Values
                          are the bending moment capacity.
                          Example:
                          { 1: 210e3,
                            2: 180e3
                          }
        :param spring: (dict) Set a rotational spring or a hinge (k=0) at node 1 or node 2.
                            { 1: k
                              2: k }

                             Set a hinged node:
                             {1: 0}

                        direction 1: ux
                        direction 2: uz
                        direction 3: phi_y
        :return element id: (int)
        """
        element_type = kwargs.get("type", "general")

        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        # add the element number
        self.count += 1

        point_1, point_2 = self._det_vertices(location_list)
        node_id1, node_id2 = self._det_node_ids(point_1, point_2)
        point_1, point_2, node_id1, node_id2, spring, mp, ai = \
            self._force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp)
        self._append_node_id(point_1, point_2, node_id1, node_id2)
        self._ensure_single_hinge(spring, node_id1, node_id2)

        # determine the length of the elements
        point = point_2 - point_1
        l = point.modulus()

        # add element
        element = Element(self.count, EA, EI, l, ai, point_1, point_2, spring)
        element.node_ids.append(node_id1)
        element.node_ids.append(node_id2)
        element.node_id1 = node_id1
        element.node_id2 = node_id2

        element.node_1 = self.node_map[node_id1]
        element.node_2 = self.node_map[node_id2]
        element.type = element_type

        self.element_map[self.count] = element

        # Register the elements per node
        for node_id in (node_id1, node_id2):
            self.node_map[node_id].elements[element.id] = element

        if mp is not None:
            self.non_linear_elements[self.count] = mp
            self.non_linear = True

        self._det_system_matrix_location(element)
        self._dead_load(g, element.id)
        return self.count

    def _det_vertices(self, location_list):
        if isinstance(location_list, Vertex_xz):
            point_1 = self._previous_point
            point_2 = location_list
        elif len(location_list) == 1:
            point_1 = self._previous_point
            point_2 = Vertex_xz(location_list[0][0], location_list[0][1])
        elif isinstance(location_list[0], (int, float)):
            point_1 = self._previous_point
            point_2 = Vertex_xz(location_list[0], location_list[1])
        elif isinstance(location_list[0], Vertex_xz):
            point_1 = location_list[0]
            point_2 = location_list[1]
        else:
            point_1 = Vertex_xz(location_list[0][0], location_list[0][1])
            point_2 = Vertex_xz(location_list[1][0], location_list[1][1])
        self._previous_point = point_2

        if self.xy_cs:
            point_1 = Vertex_xz(point_1.x, -point_1.z)
            point_2 = Vertex_xz(point_2.x, -point_2.z)
        return point_1, point_2

    def _det_node_ids(self, point_1, point_2):
        existing_node1 = False
        existing_node2 = False

        if len(self.element_map) != 0:
            count = 1
            for el in self.element_map.values():
                # check if node 1 of the element meets another node. If so, both have the same node_id
                if el.vertex_1 == point_1:
                    node_id1 = el.node_id1
                    existing_node1 = True
                elif el.vertex_2 == point_1:
                    node_id1 = el.node_id2
                    existing_node1 = True
                elif count == len(self.element_map) and existing_node1 is False:
                    self.max_node_id += 1
                    node_id1 = self.max_node_id

                # check for node 2
                if el.vertex_1 == point_2:
                    node_id2 = el.node_id1
                    existing_node2 = True
                elif el.vertex_2 == point_2:
                    node_id2 = el.node_id2
                    existing_node2 = True
                elif count == len(self.element_map) and existing_node2 is False:
                    self.max_node_id += 1
                    node_id2 = self.max_node_id

                count += 1
        else:
            node_id1 = 1
            node_id2 = 2
        return node_id1, node_id2

    @staticmethod
    def _force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp):
        """
        :return: point_1, point_2, node_id1, node_id2, spring, mp, ai
        """
        # determine the angle of the element with the global x-axis
        delta_x = point_2.x - point_1.x
        delta_z = -point_2.z - -point_1.z  # minus sign to work with an opposite z-axis
        ai = angle_x_axis(delta_x, delta_z)

        if 0.5 * math.pi < ai < 1.5 * math.pi:
            # switch points
            p = point_1
            point_1 = point_2
            point_2 = p
            delta_x = point_2.x - point_1.x
            delta_z = -point_2.z - -point_1.z  # minus sign to work with an opposite z-axis
            ai = angle_x_axis(delta_x, delta_z)

            id_ = node_id1
            node_id1 = node_id2
            node_id2 = id_

            if spring is not None:
                if 1 in spring and 2 in spring:
                    k1 = spring[1]
                    spring[1] = spring[2]
                    spring[2] = k1
                elif 1 in spring:
                    spring[2] = spring.pop(1)
                elif 2 in spring:
                    spring[1] = spring.pop(2)

            if mp is not None:
                if 1 in mp and 2 in mp:
                    m1 = mp[1]
                    mp[1] = mp[2]
                    mp[2] = m1
                elif 1 in mp:
                    mp[2] = mp.pop(1)
                elif 2 in mp:
                    mp[1] = mp.pop(2)
        return point_1, point_2, node_id1, node_id2, spring, mp, ai

    def _append_node_id(self, point_1, point_2, node_id1, node_id2):
        if node_id1 not in self.node_map:
            self.node_map[node_id1] = Node(node_id1, vertex=point_1)
        if node_id2 not in self.node_map:
            self.node_map[node_id2] = Node(node_id2, vertex=point_2)

    def _ensure_single_hinge(self, spring, node_id1, node_id2):
        if spring is not None and 0 in spring:
            """
            Must be one rotational fixed element per node. Thus keep track of the hinges (k == 0).
            """

            for node in range(1, 3):
                if spring[node] == 0:  # node is a hinged node
                    if node == 1:
                        node_id = node_id1
                    else:
                        node_id = node_id2

                    self.node_map[node_id].hinge = True

                    if len(self.node_map[node_id].elements) > 0:
                        pass_hinge = not all([el.hinge == node for el in self.node_map[node_id].elements.values()])
                    else:
                        pass_hinge = True
                    if not pass_hinge:
                        del spring[node]  # too many hinges at that element.

    def add_multiple_elements(self, location_list, n=None, dl=None):
        pass

    def _det_system_matrix_location(self, element):
        """
        :param element (Element)

         Determine the elements location in the stiffness matrix.
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

    def __assemble_system_matrix(self):
        """
        Shape of the matrix = n nodes * n d.o.f.
        Shape = n * 3
        """
        # if self.system_matrix is None:
        shape = len(self.node_map) * 3
        self.shape_system_matrix = shape
        self.system_matrix = np.zeros((shape, shape))

        for matrix_index, K in self.system_spring_map.items():
            #  first index is row, second is column
            self.system_matrix[matrix_index][matrix_index] += K

        for i in range(len(self.element_map)):
            for row_index in range(len(self.system_matrix_locations[i])):
                count = 0
                for loc in self.system_matrix_locations[i][row_index]:
                    self.system_matrix[loc[0]][loc[1]] += self.element_map[i + 1].stiffness_matrix[row_index][count]
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

        for id_, direction, force in force_list:
            # index = number of the node-1 * nd.o.f. + x, or z, or y
            # (x = 1 * i[1], y = 2 * i[1]
            index = (id_ - 1) * 3 + direction - 1
            # force = i[2]
            self.system_force_vector[index] += force
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

    def solve(self, gnl=False, force_linear=False, verbosity=0, max_iter=1000):
        """

        :param gnl: (bool) Toggle geometrical non linear calculation.
        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0. get calculation outputs. 1. silence.
        :param max_iter: (int) maximum allowed iterations.
        :return: (array) Displacements vector.
        """
        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        self.system_force_vector = None
        self._apply_perpendicular_q_load()
        self._apply_point_load()
        self._apply_moment_load()

        if gnl:
            return self._gnl(verbosity)

        if self.non_linear and not force_linear:
            return self._stiffness_adaptation(verbosity, max_iter)

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
        for el in self.element_map.values():
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

    def _stiffness_adaptation(self, verbosity, max_iter):
        self.solve(False, True)
        if verbosity == 0:
            print("Starting stiffness adaptation calculation.")

        for c in range(max_iter):
            factors = []

            # update the elements stiffnesses
            for k, v in self.non_linear_elements.items():
                el = self.element_map[k]

                for node_no, mp in v.items():
                    if node_no == 1:
                        m_e = el.node_1.Ty
                    else:
                        m_e = el.node_2.Ty

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

        return self.system_displacement_vector

    def _gnl(self, verbosity):
        div_c = 0
        last_increase = 1e3
        last_abs_u = np.abs(self.solve(False))
        if verbosity == 0:
            print("Starting geometrical non linear calculation")

        for c in range(1500):
            for el in self.element_map.values():
                delta_x = el.vertex_2.x + el.node_2.ux - el.vertex_1.x - el.node_1.ux
                # minus sign to work with an opposite z-axis
                delta_z = -el.vertex_2.z - el.node_2.uz + el.vertex_1.z + el.node_1.uz
                a_bar = angle_x_axis(delta_x, delta_z)
                l = math.sqrt(delta_x**2 + delta_z**2)
                ai = a_bar + el.node_1.phi_y
                aj = a_bar + el.node_2.phi_y
                el.compile_kinematic_matrix(ai, aj, l)
                #el.compile_constitutive_matrix(el.EA, el.EI, l)
                el.compile_stiffness_matrix()

            current_abs_u = np.abs(self.solve(gnl=False, verbosity=1))
            global_increase = np.sum(current_abs_u) / np.sum(last_abs_u)

            if global_increase < 0.1:
                print("Divergence in {} iterations".format(c))
                break
            if global_increase - 1 < 1e-8 and global_increase >= 1:
                print("Convergence in {} iterations".format(c))
                break
            if global_increase - last_increase > 0 and global_increase > 1:
                div_c += 1

                if div_c > 25:
                    print("Divergence in {} iterations".format(c))
                    break

            last_abs_u = current_abs_u
            last_increase = global_increase

    def _support_check(self, node_id):
        if self.node_map[node_id].hinge:
            raise Exception ("You cannot add a support to a hinged node.")

    def add_support_hinged(self, node_id):
        """
        adds a hinged support a the given node
        :param node_id: integer representing the nodes ID
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self.set_displacement_vector([(id_, 1), (id_, 2)])

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_roll(self, node_id, direction=2):
        """
        adds a rollable support at the given node
        :param node_id: integer representing the nodes ID
        :param direction: integer representing the direction that is fixed: x = 1, z = 2
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self.set_displacement_vector([(id_, direction)])

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction)

    def add_support_fixed(self, node_id):
        """
        adds a fixed support at the given node
        :param node_id: (int/ list) integer representing the nodes ID
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self.set_displacement_vector([(id_, 1), (id_, 2), (id_, 3)])

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def add_support_spring(self, node_id, translation, k, roll=False):
        """
        :param translation: (int) Integer representing prevented translation.
        1 = translation in x
        2 = translation in z
        3 = rotation in y
        :param node_id: (int/ list) Integer representing the nodes ID.
        :param k: (flt) Stiffness of the spring
        :param roll: If set to True, only the translation of the spring is controlled.

        The stiffness of the spring is added in the system matrix at the location that represents the node and the
        displacement.
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)

            # determine the location in the system matrix
            # row and column are the same
            matrix_index = (id_ - 1) * 3 + translation - 1

            self.system_spring_map[matrix_index] = k

            # add the support to the support list for the plotter
            if translation == 1:
                self.supports_spring_x.append(self.node_map[id_])
            elif translation == 2:
                self.supports_spring_z.append(self.node_map[id_])
            else:
                self.supports_spring_y.append(self.node_map[id_])

            if not roll:  # fix the other d.o.f.
                if translation == 1:  # translation spring in x-axis
                    self.set_displacement_vector([(id_, 2)])
                elif translation == 2:  # translation spring in z-axis
                    self.set_displacement_vector([(id_, 1)])
                elif translation == 3:  # rotational spring in y-axis
                    self.set_displacement_vector([(id_, 1), (id_, 2)])

    def _dead_load(self, g, element_id):
        self.loads_dead_load.append((element_id, g))
        self.element_map[element_id].dead_load = g

    def q_load(self, q, element_id, direction="element"):
        """
        :param element_id: (int) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"
        :return: (None)
        """
        if not isinstance(element_id, (tuple, list)):
            element_id = (element_id,)
            q = (q,)

        for i in range(len(element_id)):
            self.plotter.max_q = max(self.plotter.max_q, q[i])
            self.loads_q[element_id[i]] = q[i]
            el = self.element_map[element_id[i]]
            el.q_load = q[i]
            el.q_direction = direction

    def _apply_perpendicular_q_load(self):
        for element_id, g in self.loads_dead_load:
            element = self.element_map[element_id]
            q_perpendicular = element.all_q_load

            if q_perpendicular == 0:
                continue
            elif element.q_direction == "x" or element.q_direction == "y" or element.dead_load:
                self._apply_parallel_q_load(element)

            q_perpendicular *= self.load_factor
            kl = element.constitutive_matrix[1][1] * 1e6
            kr = element.constitutive_matrix[2][2] * 1e6
            # minus because of systems positive rotation
            left_moment = -det_moment(kl, kr, q_perpendicular, 0, element.EI, element.l)
            right_moment = det_moment(kl, kr, q_perpendicular, element.l, element.EI, element.l)

            rleft = det_shear(kl, kr, q_perpendicular, 0, element.EI, element.l)
            rright = -det_shear(kl, kr, q_perpendicular, element.l, element.EI, element.l)

            rleft_x = rleft * math.sin(element.ai)
            rright_x = rright * math.sin(element.ai)

            rleft_z = rleft * math.cos(element.ai)
            rright_z = rright * math.cos(element.ai)

            # place the primary forces in the 6*6 primary force matrix
            element.element_primary_force_vector[0] -= rleft_x
            element.element_primary_force_vector[1] -= rleft_z
            element.element_primary_force_vector[2] += left_moment
            element.element_primary_force_vector[3] -= rright_x
            element.element_primary_force_vector[4] -= rright_z
            element.element_primary_force_vector[5] += right_moment

            # system force vector. System forces = element force * -1
            # first row are the moments
            # second and third row are the reaction forces. The direction
            self.set_force_vector([(element.node_1.id, 3, -left_moment), (element.node_2.id, 3, -right_moment),
                                   (element.node_1.id, 2, rleft_z), (element.node_2.id, 2, rright_z),
                                   (element.node_1.id, 1, rleft_x), (element.node_2.id, 1, rright_x)])

    def _apply_parallel_q_load(self, element):
        direction = element.q_direction
        if direction == "x":
            factor = abs(math.cos(element.ai))
        elif direction == "y":
            factor = abs(math.sin(element.ai))
        else:  # element has got only dead load
            factor = abs(math.sin(element.ai))

        # q_load working at parallel to the elements x-axis
        q_element = (element.q_load + element.dead_load) * factor * self.load_factor

        Fx = q_element * math.cos(element.ai) * element.l * 0.5
        Fz = q_element * math.sin(element.ai) * element.l * -0.5

        element.element_primary_force_vector[0] -= Fx
        element.element_primary_force_vector[1] -= Fz
        element.element_primary_force_vector[3] -= Fx
        element.element_primary_force_vector[4] -= Fz

        self.set_force_vector([
            (element.node_1.id, 2, Fz), (element.node_2.id, 2, Fz),
            (element.node_1.id, 1, Fx), (element.node_2.id, 1, Fx)])

    def point_load(self, node_id, Fx=0, Fz=0):
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)
            Fx = (Fx,)
            Fz = (Fz,)
        elif Fx == 0:
            Fx = [0 for _ in Fz]
        elif Fz == 0:
            Fz = [0 for _ in Fx]

        for i in range(len(node_id)):
            self.plotter.max_system_point_load = max(self.plotter.max_system_point_load, (Fx[i]**2 + Fz[i]**2)**0.5)
            self.loads_point.append((node_id[i], Fx[i], Fz[i]))

    def _apply_point_load(self):
        for node_id, Fx, Fz in self.loads_point:
            # system force vector.
            self.set_force_vector([(node_id, 1, Fx * self.load_factor), (node_id, 2, Fz * self.load_factor)])

    def moment_load(self, node_id, Ty):
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)
            Ty = (Ty,)

        for i in range(len(node_id)):
            self.loads_moment.append((node_id[i], 3, Ty[i]))

    def _apply_moment_load(self):
        for node_id, _, Ty in self.loads_moment:
            self.set_force_vector([(node_id, 3, Ty * self.load_factor)])

    def show_structure(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True, supports=True):
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.plot_structure(figsize, verbosity, show, supports, scale, offset)

    def show_bending_moment(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.bending_moment(factor, figsize, verbosity, scale, offset, show)

    def show_axial_force(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    def show_shear_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.shear_force(figsize, verbosity, scale, offset, show)

    def show_reaction_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        figsize = self.figsize if figsize is None else figsize
        self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                          linear=False):
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
                    "alpha": el.ai,
                    "u": el.extension[0],
                    "N_1": el.N_1,
                    "N_2": el.N_2
                }
            else:
                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.ai,
                    "u": el.extension[0],
                    "N_1": el.N_1,
                    "N_2": el.N_2,
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
            for el in self.element_map.values():

                if el.type == "truss":
                    result_list.append({
                        "id": el.id,
                        "length": el.l,
                        "alpha": el.ai,
                        "u": el.extension[0],
                        "N_1": el.N_1,
                        "N_2": el.N_2
                    }
                    )

                else:
                    result_list.append(
                        {
                            "id": el.id,
                            "length": el.l,
                            "alpha": el.ai,
                            "u": el.extension[0],
                            "N_1": el.N_1,
                            "N_2": el.N_2,
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

    def get_element_result_range(self, unit):
        if unit == "shear":
            return [el.shear_force[0] for el in self.element_map.values()]

    def find_node_id(self, vertex):
        """
        :param vertex: (Vertex/ list/ tpl) Vertex_xz, [x, z], (x, z)
        :return: (int/ None) id of the node at the location of the vertex
        """
        if isinstance(vertex, (list, tuple)):
            vertex = Vertex_xz(vertex)
        try:
            a = -1 if self.xy_cs else 1
            tol = 1e-9
            return next(filter(lambda x: math.isclose(x.vertex.x, vertex.x, abs_tol=tol)
                               and math.isclose(x.vertex.z, vertex.z * a, abs_tol=tol),
                               self.node_map.values())).id
        except StopIteration:
            return None

    def nodes_range(self, coordinate):
        """
        Retrieve a list with coordinates x or z (y).

        :param coordinate: (str) "both", 'x', 'y' or 'z'
        :return: (list)
        """
        return list(
            map(
                lambda x: x.vertex.x if coordinate == 'x'
                else x.vertex.z if coordinate == 'z' else -x.vertex.z if coordinate == 'y'
                else None,
                self.node_map.values()))