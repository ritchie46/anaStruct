import math
import numpy as np
from anastruct.basic import converge, angle_x_axis, FEMException
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.elements import Element, det_moment, det_shear
from anastruct.fem.node import Node
from anastruct.vertex import Vertex
from anastruct.fem.plotter import Plotter
from anastruct.fem.elements import geometric_stiffness_matrix


class SystemElements:
    """
    Modelling any structure starts with an object of this class.
    """

    def __init__(self, figsize=(12, 8), EA=15e3, EI=5e3, load_factor=1, mesh=50, plot_backend='mpl'):
        """
        E = Young's modulus
        A = Area
        I = Moment of Inertia

        :param figsize: (tpl) Set the standard plotting size.
        :param EA: (flt) Standard E * A. Set the standard values of EA if none provided when generating an element.
        :param EI: (flt) Standard E * I. Set the standard values of EA if none provided when generating an element.
        :param load_factor: (flt) Multiply all loads with this factor.
        :param mesh: (int) Plotting mesh. Has no influence on the calculation.
        :param plot_backend: (str)  matplotlib  -> "mpl"  (Currently only the matplotlib one is stable)
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
        self.orientation_cs = -1  # needed for the loads directions

        # structure system
        self.element_map = {}  # maps element ids to the Element objects.
        self.node_map = {}  # maps node ids to the Node objects.
        self.node_element_map = {}  # maps node ids to Element objects
        self.system_spring_map = {}  # keys matrix index (for both row and columns), values K

        # list of indexes that remain after conditions are applied
        self._remainder_indexes = []

        # keep track of the node_id of the supports
        self.supports_fixed = []
        self.supports_hinged = []
        self.supports_roll = []
        self.supports_spring_x = []
        self.supports_spring_z = []
        self.supports_spring_y = []
        self.supports_roll_direction = []

        # keep track of the loads
        self.loads_point = {}  # node ids with a point loads
        self.loads_q = {}  # element ids with a q-load
        self.loads_moment = {}
        self.loads_dead_load = []  # element ids with q-load due to dead load

        # results
        self.reaction_forces = {}  # node objects
        self.non_linear = False
        self.non_linear_elements = {}  # keys are element ids, values are dicts: {node_index: max moment capacity}

        # previous point of element
        self._previous_point = Vertex(0, 0)
        self.load_factor = load_factor

        # Objects state
        self.count = 0
        self.system_matrix_locations = []
        self.system_matrix = None
        self.system_force_vector = None
        self.system_displacement_vector = None
        self.shape_system_matrix = None
        self.reduced_force_vector = None
        self.reduced_system_matrix = None
        self._vertices = {}  # maps vertices to node ids

    def add_element_grid(self, x, y, EA=None, EI=None, g=None, mp=None, spring=None, **kwargs):
        a = np.ones(x.shape)
        if EA is None:
            EA = a * self.EA
        if EI is None:
            EI = a * self.EI
        if g is None:
            g = a * 0

        for i in range(len(x) - 1):
            self.add_element([[x[i], y[i]], [x[i + 1], y[i + 1]]], EA[i], EI[i], g[i], mp, spring, **kwargs)

    def add_truss_element(self, location, EA=None):
        """
        .. highlight:: python

        Add an element that only has axial force.

        :param location: (list/ Vertex) The two nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        :param EA: (flt) EA
        :return: (int) Elements ID.
        """
        return self.add_element(location, EA, 1e-14, element_type='truss')

    def add_element(self, location, EA=None, EI=None, g=0, mp=None, spring=None, **kwargs):
        """
        :param location: (list/ Vertex) The two nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        :param EA: (flt) EA
        :param EI: (flt) EI
        :param g: (flt) Weight per meter. [kN/m] / [N/m]
        :param mp: (dict) Set a maximum plastic moment capacity. Keys are integers representing the nodes. Values
                          are the bending moment capacity.

            :Example:

            .. code-block:: python

                mp={1: 210e3,
                    2: 180e3}

        :param spring: (dict) Set a rotational spring or a hinge (k=0) at node 1 or node 2.

            :Example:

            .. code-block:: python

                spring={1: k
                        2: k}


                # Set a hinged node:
                spring={1: 0}


        :return: (int) Elements ID.
        """
        element_type = kwargs.get("element_type", "general")

        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        # add the element number
        self.count += 1

        point_1, point_2 = self._det_vertices(location)
        node_id1, node_id2 = self._det_node_ids(point_1, point_2)

        point_1, point_2, node_id1, node_id2, spring, mp, ai = \
            self._force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp)

        self._append_node_id(point_1, point_2, node_id1, node_id2)
        self._ensure_single_hinge(spring, node_id1, node_id2)

        # add element
        element = Element(self.count, EA, EI, (point_2 - point_1).modulus(), ai, point_1, point_2, spring)
        element.node_id1 = node_id1
        element.node_id2 = node_id2
        element.node_map = {node_id1: self.node_map[node_id1],
                            node_id2: self.node_map[node_id2]}

        element.type = element_type

        self.element_map[self.count] = element

        for node in (node_id1, node_id2):
            if node in self.node_element_map:
                self.node_element_map[node].append(element)
            else:
                self.node_element_map[node] = [element]

        # Register the elements per node
        for node_id in (node_id1, node_id2):
            self.node_map[node_id].elements[element.id] = element

        if mp is not None:
            assert type(mp) == dict, "The mp parameter should be a dictionary."
            self.non_linear_elements[self.count] = mp
            self.non_linear = True
        self._dead_load(g, element.id)

        return self.count

    def _det_vertices(self, location_list):
        if isinstance(location_list, Vertex):
            point_1 = self._previous_point
            point_2 = Vertex(location_list)
        elif len(location_list) == 1:
            point_1 = self._previous_point
            point_2 = Vertex(location_list[0][0], location_list[0][1])
        elif isinstance(location_list[0], (int, float)):
            point_1 = self._previous_point
            point_2 = Vertex(location_list[0], location_list[1])
        elif isinstance(location_list[0], Vertex):
            point_1 = location_list[0]
            point_2 = location_list[1]
        else:
            point_1 = Vertex(location_list[0][0], location_list[0][1])
            point_2 = Vertex(location_list[1][0], location_list[1][1])
        self._previous_point = point_2

        return point_1, point_2

    def _det_node_ids(self, point_1, point_2):
        node_ids = []
        for p in (point_1, point_2):
            k = str(p)
            if k in self._vertices:
                node_id = self._vertices[k]
            else:
                node_id = len(self._vertices) + 1
                self._vertices[k] = node_id
            node_ids.append(node_id)
        return node_ids

    @staticmethod
    def _force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp):
        """
        Forces the elements to be in the first and the last quadrant of the unity circle. Meaning the first node is
        always left and the last node is always right. Or the are both on one vertical line.

        The angle of the element will thus always be between -90 till +90 degrees.
        :return: point_1, point_2, node_id1, node_id2, spring, mp, ai
        """
        # determine the angle of the element with the global x-axis
        delta_x = point_2.x - point_1.x
        delta_z = point_2.z - point_1.z  # minus sign to work with an opposite z-axis
        ai = -angle_x_axis(delta_x, delta_z)

        if delta_x < 0:
            # switch points
            point_1, point_2 = point_2, point_1
            node_id1, node_id2 = node_id2, node_id1

            ai = -angle_x_axis(-delta_x, -delta_z)

            if spring is not None:
                assert type(spring) == dict, "The spring parameter should be a dictionary."
                if 1 in spring and 2 in spring:
                    spring[1], spring[2] = spring[2], spring[1]
                elif 1 in spring:
                    spring[2] = spring.pop(1)
                elif 2 in spring:
                    spring[1] = spring.pop(2)

            if mp is not None:
                if 1 in mp and 2 in mp:
                    mp[1], mp[2] = mp[2], mp[1]
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

    def add_multiple_elements(self, location, n=None, dl=None, EA=None, EI=None, g=0, mp=None, spring=None,
                              **kwargs):
        """
        Add multiple elements defined by the first and the last point.

        :param location: See 'add_element' method
        :param n: (int) Number of elements.
        :param dl: (flt) Distance between the elements nodes.
        :param EA: See 'add_element' method
        :param EI: See 'add_element' method
        :param g: See 'add_element' method
        :param mp: See 'add_element' method
        :param spring: See 'add_element' method

        **Keyword Args:**

        :param element_type: (str) See 'add_element' method
        :param first: (dict) Different arguments for the first element
        :param last: (dict) Different arguments for the last element

            :Example:

            .. code-block:: python

                last={'EA': 1e3, 'mp': 290}

        :return: (list) Element IDs
        """

        first = kwargs.get("first", {})
        last = kwargs.get("last", {})
        element_type = kwargs.get("element_type", "general")

        for el in (first, last):
            if "EA" not in el:
                el["EA"] = EA
            if "EI" not in el:
                el["EI"] = EI
            if "g" not in el:
                el["g"] = g
            if "mp" not in el:
                el["mp"] = mp
            if "spring" not in el:
                el["spring"] = spring
            if "element_type" not in el:
                el["element_type"] = element_type

        point_1, point_2 = self._det_vertices(location)
        length = (point_2 - point_1).modulus()
        direction = (point_2 - point_1).unit()

        if type(n) == type(dl):
            raise FEMException("Wrong parameters", "One, and only one, of n and dl should be passed as argument.")
        elif n:
            dl = length / n

        point = point_1 + direction * dl
        elements = [self.add_element((point_1, point), first["EA"], first["EI"], first["g"], first["mp"], first["spring"],
                         element_type=first["element_type"])]

        l = 2 * dl
        while l < length:
            point += direction * dl
            elements.append(self.add_element(point, EA, EI, g, mp, spring, element_type=element_type))
            l += dl

        elements.append(self.add_element(point_2, last["EA"], last["EI"], last["g"], last["mp"], last["spring"],
                         element_type=last["element_type"]))
        return elements

    def __assemble_system_matrix(self, validate=False, geometric_matrix=False):
        """
        Shape of the matrix = n nodes * n d.o.f.
        Shape = n * 3
        """
        if not geometric_matrix:
            shape = len(self.node_map) * 3
            self.shape_system_matrix = shape
            self.system_matrix = np.zeros((shape, shape))

        for matrix_index, K in self.system_spring_map.items():
            #  first index is row, second is column
            self.system_matrix[matrix_index][matrix_index] += K

        # Determine the elements location in the stiffness matrix.
        # system matrix [K]
        #
        # [fx 1] [K         |  \ node 1 starts at row 1
        # |fz 1] |  K       |  /
        # |Ty 1] |   K      | /
        # |fx 2] |    K     |  \ node 2 starts at row 4
        # |fz 2] |     K    |  /
        # |Ty 2] |      K   | /
        # |fx 3] |       K  |  \ node 3 starts at row 7
        # |fz 3] |        K |  /
        # [Ty 3] [         K] /
        #
        #         n   n  n
        #         o   o  o
        #         d   d  d
        #         e   e  e
        #         1   2  3
        #
        # thus with appending numbers in the system matrix: column = row

        for i in range(len(self.element_map)):
            element = self.element_map[i + 1]
            if geometric_matrix:
                element_matrix = geometric_stiffness_matrix(element.l, element.N_1, element.ai)
            else:
                element_matrix = element.stiffness_matrix

            # n1 and n2 are starting indexes of the rows and the columns for node 1 and node 2
            n1 = (element.node_1.id - 1) * 3
            n2 = (element.node_2.id - 1) * 3
            self.system_matrix[n1: n1 + 3, n1: n1 + 3] += element_matrix[0:3, :3]
            self.system_matrix[n1: n1 + 3, n2: n2 + 3] += element_matrix[0:3, 3:]

            self.system_matrix[n2: n2 + 3, n1: n1 + 3] += element_matrix[3:6, :3]
            self.system_matrix[n2: n2 + 3, n2: n2 + 3] += element_matrix[3:6, 3:]

        # returns True if symmetrical.
        if validate:
            assert np.allclose((self.system_matrix.transpose()), self.system_matrix)

    def _set_force_vector(self, force_list):
        """
        :param force_list: list containing tuples with the
        1. number of the node,
        2. the number of the direction (1 = x, 2 = z, 3 = y)
        3. the force
        [(1, 3, 1000)] node=1, direction=3 (y), force=1000
        list may contain multiple tuples
        :return: Vector with forces on the nodes
        """
        for id_, direction, force in force_list:
            self.system_force_vector[(id_ - 1) * 3 + direction - 1] += force

        return self.system_force_vector

    def _set_displacement_vector(self, nodes_list):
        """
        :param nodes_list: list containing tuples with
        1.the node
        2. the d.o.f. that is set
        :return: Vector with the displacements of the nodes (If displacement is not known, the value is set
        to NaN)
        """
        if self.system_displacement_vector is None:
            self.system_displacement_vector = np.empty(len(self._vertices) * 3)
            self.system_displacement_vector[:] = np.NaN

        for i in nodes_list:
            index = (i[0] - 1) * 3 + i[1] - 1
            self.system_displacement_vector[index] = 0
        return self.system_displacement_vector

    def __process_conditions(self):
        indexes = []
        # remove the unsolvable values from the matrix and vectors
        for i in range(self.shape_system_matrix):
            if self.system_displacement_vector[i] == 0:
                indexes.append(i)
            else:
                self._remainder_indexes.append(i)

        self.system_displacement_vector = np.delete(self.system_displacement_vector, indexes, 0)
        self.reduced_force_vector = np.delete(self.system_force_vector, indexes, 0)
        self.reduced_system_matrix = np.delete(self.system_matrix, indexes, 0)
        self.reduced_system_matrix = np.delete(self.reduced_system_matrix, indexes, 1)

    def solve(self, force_linear=False, verbosity=0, max_iter=200, **kwargs):

        """
        Compute the results of current model.

        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0. Log calculation outputs. 1. silence.
        :param max_iter: (int) Maximum allowed iterations.

        :return: (array) Displacements vector.
        """

        # kwargs: arguments for the iterative solver callers such as the _stiffness_adaptation method.
        #                naked (bool) Default = False, if True force lines won't be computed.

        naked = kwargs.get("naked", False)
        gnl = kwargs.get("gnl", False)

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()

        self.system_force_vector = self.system_force_vector = np.zeros(len(self._vertices) * 3)
        self._apply_perpendicular_q_load()
        self._apply_point_load()
        self._apply_moment_load()

        if self.non_linear and not force_linear:
            return self._stiffness_adaptation(verbosity, max_iter)

        assert(self.system_force_vector is not None), "There are no forces on the structure"
        self._remainder_indexes = []
        self.__assemble_system_matrix()
        if gnl:
            self.__assemble_system_matrix(geometric_matrix=True)
        self.__process_conditions()

        # solution of the reduced system (reduced due to support conditions)
        reduced_displacement_vector = np.linalg.solve(self.reduced_system_matrix, self.reduced_force_vector)

        # add the solution of the reduced system in the complete system displacement vector
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        np.put(self.system_displacement_vector, self._remainder_indexes, reduced_displacement_vector)

        # determine the displacement vector of the elements
        for el in self.element_map.values():
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

            # node 1 ux, uz, phi
            el.element_displacement_vector[:3] = self.system_displacement_vector[index_node_1: index_node_1 + 3]
            # node 2 ux, uz, phi
            el.element_displacement_vector[3:] = self.system_displacement_vector[index_node_2: index_node_2 + 3]
            el.determine_force_vector()

        if not naked:
            # determining the node results in post processing class
            self.post_processor.node_results_elements()
            self.post_processor.node_results_system()
            self.post_processor.reaction_forces()
            self.post_processor.element_results()

            # check the values in the displacement vector for extreme values, indicating a flawed calculation
            assert(np.any(self.system_displacement_vector < 1e6)), "The displacements of the structure exceed 1e6. " \
                                                                   "Check your support conditions," \
                                                                   "or your elements Young's modulus"

        return self.system_displacement_vector

    def _stiffness_adaptation(self, verbosity, max_iter):
        self.solve(True, naked=True)
        if verbosity == 0:
            print("Starting stiffness adaptation calculation.")

        # check validity
        assert all([mp > 0 for mpd in self.non_linear_elements.values() for mp in mpd]), \
            "Cannot solve for an mp = 0. If you want a hinge set the spring stiffness equal to 0."

        for c in range(max_iter):
            factors = []

            # update the elements stiffnesses
            for k, v in self.non_linear_elements.items():
                el = self.element_map[k]

                for node_no, mp in v.items():
                    if node_no == 1:
                        # Fast Ty
                        m_e = el.element_force_vector[2] + el.element_primary_force_vector[2]
                    else:
                        # Fast Ty
                        m_e = el.element_force_vector[5] + el.element_primary_force_vector[5]

                    if abs(m_e) > mp:
                        el.nodes_plastic[node_no - 1] = True
                    if el.nodes_plastic[node_no - 1]:
                        factor = converge(m_e, mp, 3)
                        factors.append(factor)
                        el.update_stiffness(factor, node_no)

            if not np.allclose(factors, 1, 1e-3):
                self.solve(force_linear=True, naked=True)
            else:
                self.post_processor.node_results_elements()
                self.post_processor.node_results_system()
                self.post_processor.reaction_forces()
                self.post_processor.element_results()
                break

        if c == max_iter - 1:
            print("Couldn't solve the in the amount of iterations given.")
        elif verbosity == 0:
            print("Solved in {} iterations".format(c))
        return self.system_displacement_vector

    def gnl(self, verbosity=0):
        if verbosity == 0:
            print("Starting geometrical non linear calculation")

        last_abs_u = np.abs(self.solve(force_linear=True))

        for c in range(200):
            current_abs_u = np.abs(self.solve(gnl=True))

            global_increase = np.sum(current_abs_u) / np.sum(last_abs_u)
            last_abs_u = current_abs_u

            if math.isclose(global_increase, 1.0):
                break

        if verbosity == 0:
            print("Solved in {} iterations".format(c))

    def _support_check(self, node_id):
        if self.node_map[node_id].hinge:
            raise FEMException ("Flawed inputs", "You cannot add a support to a hinged node.")

    def add_support_hinged(self, node_id):
        """
        Model a hinged support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self._set_displacement_vector([(id_, 1), (id_, 2)])

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_roll(self, node_id, direction=2):
        """
        Adds a rolling support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        :param direction: (int/ list) Represents the direction that is fixed: x = 1, y = 2
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self._set_displacement_vector([(id_, direction)])

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction)

    def add_support_fixed(self, node_id):
        """
        Add a fixed support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)

        for id_ in node_id:
            self._support_check(id_)
            self._set_displacement_vector([(id_, 1), (id_, 2), (id_, 3)])

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def add_support_spring(self, node_id, translation, k, roll=False):
        """
        Add a translational support at a given node.

        :param translation: (int/ list) Represents the prevented translation.

            **Note**

            |  1 = translation in x
            |  2 = translation in z
            |  3 = rotation in y

        :param node_id: (int/ list) Integer representing the nodes ID.
        :param k: (flt) Stiffness of the spring
        :param roll: (bool) If set to True, only the translation of the spring is controlled.

        """
        # The stiffness of the spring is added in the system matrix at the location that represents the node and the
        # displacement.
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
                    self._set_displacement_vector([(id_, 2)])
                elif translation == 2:  # translation spring in z-axis
                    self._set_displacement_vector([(id_, 1)])
                elif translation == 3:  # rotational spring in y-axis
                    self._set_displacement_vector([(id_, 1), (id_, 2)])

    def _dead_load(self, g, element_id):
        self.loads_dead_load.append((element_id, g))
        self.element_map[element_id].dead_load = g

    def q_load(self, q, element_id, direction="element"):
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"

        """
        if not isinstance(element_id, (tuple, list)):
            element_id = (element_id,)
            q = (q,)

        if type(q) != type(element_id):  # arguments should be q, [id_1, [id_2]
            q = [q for _ in element_id]

        for i in range(len(element_id)):
            self.plotter.max_q = max(self.plotter.max_q, abs(q[i]))
            self.loads_q[element_id[i]] = q[i] * self.orientation_cs * self.load_factor
            el = self.element_map[element_id[i]]
            el.q_load = q[i] * self.orientation_cs * self.load_factor
            el.q_direction = direction

    def _apply_perpendicular_q_load(self):
        for element_id, g in self.loads_dead_load:
            element = self.element_map[element_id]
            q_perpendicular = element.all_q_load

            if q_perpendicular == 0:
                continue
            elif element.q_direction == "x" or element.q_direction == "y" or element.dead_load:
                self._apply_parallel_q_load(element)

            kl = element.constitutive_matrix[1][1] * 1e6
            kr = element.constitutive_matrix[2][2] * 1e6

            if math.isclose(kl, kr):
                left_moment = det_moment(kl, kr, q_perpendicular, 0, element.EI, element.l)
                right_moment = -left_moment
                rleft = det_shear(kl, kr, q_perpendicular, 0, element.EI, element.l)
                rright = rleft
            else:
                # minus because of systems positive rotation
                left_moment = det_moment(kl, kr, q_perpendicular, 0, element.EI, element.l)
                right_moment = -det_moment(kl, kr, q_perpendicular, element.l, element.EI, element.l)
                rleft = det_shear(kl, kr, q_perpendicular, 0, element.EI, element.l)
                rright = -det_shear(kl, kr, q_perpendicular, element.l, element.EI, element.l)

            rleft_x = rleft * math.sin(element.ai)
            rright_x = rright * math.sin(element.ai)

            rleft_z = rleft * math.cos(element.ai)
            rright_z = rright * math.cos(element.ai)

            primary_force = np.array([rleft_x, rleft_z, left_moment, rright_x, rright_z, right_moment])
            element.element_primary_force_vector -= primary_force

            # Set force vector
            self.system_force_vector[(element.node_id1 - 1) * 3: (element.node_id1 - 1) * 3 + 3] += primary_force[0:3]
            self.system_force_vector[(element.node_id2 - 1) * 3: (element.node_id2 - 1) * 3 + 3] += primary_force[3:]

    def _apply_parallel_q_load(self, element):
        direction = element.q_direction
        # dead load
        factor_dl = abs(math.sin(element.ai))

        def update(Fx, Fz):
            element.element_primary_force_vector[0] -= Fx
            element.element_primary_force_vector[1] -= Fz
            element.element_primary_force_vector[3] -= Fx
            element.element_primary_force_vector[4] -= Fz

            self._set_force_vector([
                (element.node_1.id, 2, Fz), (element.node_2.id, 2, Fz),
                (element.node_1.id, 1, Fx), (element.node_2.id, 1, Fx)])

        if direction == "x":
            factor = abs(math.cos(element.ai))

            for q_element in (element.q_load * factor, element.dead_load * factor_dl):
                # q_load working at parallel to the elements x-axis          # set the proper direction
                Fx = -q_element * math.cos(element.ai) * element.l * 0.5
                Fz = q_element * abs(math.sin(element.ai)) * element.l * 0.5 * np.sign(math.sin(element.ai))

                update(Fx, Fz)

        else:
            if math.isclose(element.ai, 0):
                # horizontal element cannot have parallel forces due to self weight or q-load in y direction.
                return None
            factor = abs(math.sin(element.ai))

            for q_element in (element.q_load * factor, element.dead_load * factor_dl):
                # q_load working at parallel to the elements x-axis          # set the proper direction
                Fx = q_element * math.cos(element.ai) * element.l * 0.5 * -np.sign(math.sin(element.ai))
                Fz = q_element * abs(math.sin(element.ai)) * element.l * 0.5

                update(Fx, Fz)

    def point_load(self, node_id, Fx=0, Fz=0):
        """
        Apply a point load to a node.

        :param node_id: (int) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fz: (flt/ list) Force in global x direction.
        """
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
            self.loads_point[node_id[i]] = (Fx[i], Fz[i] * self.orientation_cs)

    def _apply_point_load(self):
        for node_id in self.loads_point:
            Fx, Fz = self.loads_point[node_id]
            # system force vector.
            self._set_force_vector([(node_id, 1, Fx * self.load_factor), (node_id, 2, Fz * self.load_factor)])

    def moment_load(self, node_id, Ty):
        """
        Apply a moment on a node.

        :param node_id: (int) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        if not isinstance(node_id, (tuple, list)):
            node_id = (node_id,)
            Ty = (Ty,)

        for i in range(len(node_id)):
            self.loads_moment[node_id[i]] = Ty[i]

    def _apply_moment_load(self):
        for node_id, Ty in self.loads_moment.items():
            self._set_force_vector([(node_id, 3, Ty * self.load_factor)])

    def show_structure(self, verbosity=0, scale=1., offset=(0, 0), figsize=None, show=True, supports=True):
        """
        Plot the structure.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param supports: (bool) Show the supports.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.plot_structure(figsize, verbosity, show, supports, scale, offset)

    def show_bending_moment(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot the bending moment.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.bending_moment(factor, figsize, verbosity, scale, offset, show)

    def show_axial_force(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot the axial force.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    def show_shear_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot the shear force.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.shear_force(figsize, verbosity, scale, offset, show)

    def show_reaction_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot the reaction force.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                          linear=False):
        """
        Plot the displacement.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.displacements(factor, figsize, verbosity, scale, offset, show, linear)

    def show_results(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot all the results in one window.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        self.plotter.results_plot(figsize, verbosity, scale, offset, show)

    def get_node_results_system(self, node_id=0):
        """
        These are the node results. These are the opposite of the forces and displacements working on the elements and
        may seem counter intuitive.

        :param node_id: (integer) representing the node's ID. If integer = 0, the results of all nodes are returned
        :return:

        |  if node_id == 0: (list)
        |
        |    Returns a list containing tuples with the results:
        |
        |     [(id, Fx, Fz, Ty, ux, uy, phi_y), (id, Fx, Fz...), () .. ]
        |
        |  if node_id > 0: (dict)
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
                "uy": -node.uz,
                "phi_y": node.phi_y
            }
        else:
            for node in self.node_map.values():
                result_list.append((node.id, node.Fx, node.Fz, node.Ty, node.ux, -node.uz, node.phi_y))
        return result_list

    def get_node_displacements(self, node_id=0):
        """
        :param node_id: (int) Represents the node's ID. If integer = 0, the results of all nodes are returned.
        :return:
        |  if node_id == 0: (list)
        |
        |    Returns a list containing tuples with the results:
        |
        |     [(id, Fx, Fz, Ty, ux, uy, phi_y), (id, Fx, Fz...), () .. ]
        |
        |  if node_id > 0: (dict)
        """
        result_list = []
        if node_id != 0:
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "ux": -node.ux,
                "uy": node.uz,  # - * -  = +
                "phi_y": node.phi_y
            }
        else:
            for node in self.node_map.values():
                result_list.append((node.id, -node.ux, node.uz, node.phi_y))
        return result_list

    def get_element_results(self, element_id=0, verbose=False):
        """
        :param element_id: (int) representing the elements ID. If elementID = 0 the results of all elements are returned.
        :param verbose: (bool) If set to True the numerical results for the deflection and the bending moments are
                               returned.

        :return:
        |
        |  if node_id == 0: (list)
        |
        |    Returns a list containing tuples with the results:
        |
        |     [(id, Fx, Fz, Ty, ux, uy, phi_y), (id, Fx, Fz...), () .. ]
        |
        |  if node_id > 0: (dict)
        |

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
        """
        Useful when added lots of elements. Returns a list of all the queried unit.

        :param unit: (str)
            - 'shear'
            - 'moment'
            - 'axial'

        :return: (list)
        """
        if unit == "shear":
            return [el.shear_force[0] for el in self.element_map.values()]
        elif unit == "moment":
            return [el.bending_moment[0] for el in self.element_map.values()]
        elif unit == "axial":
            return [el.N_1 for el in self.element_map.values()]

    def get_node_result_range(self, unit):
        """
        Query a list with node results.

           :param unit: (str)
            - 'uy'
            - 'ux'
        :return: (list)
        """
        if unit == "uy":
            return [node.uz for node in self.node_map.values()]   # - * -  = +
        elif unit == "ux":
            return [-node.ux for node in self.node_map.values()]

    def find_node_id(self, vertex):
        """
        Retrieve the ID of a certain location.

        :param vertex: (Vertex/ list/ tpl) Vertex_xz, [x, y], (x, y)
        :return: (int/ None) id of the node at the location of the vertex
        """
        if isinstance(vertex, (list, tuple)):
            vertex = Vertex(vertex)
        try:
            tol = 1e-9
            return next(filter(lambda x: math.isclose(x.vertex.x, vertex.x, abs_tol=tol)
                               and math.isclose(x.vertex.y, vertex.y, abs_tol=tol),
                               self.node_map.values())).id
        except StopIteration:
            return None

    def nodes_range(self, dimension):
        """
        Retrieve a list with coordinates x or z (y).

        :param dimension: (str) "both", 'x', 'y' or 'z'
        :return: (list)
        """
        return list(
            map(
                lambda x: x.vertex.x if dimension == 'x'
                else x.vertex.z if dimension == 'z' else x.vertex.y if dimension == 'y'
                else None,
                self.node_map.values()))

    def nearest_node(self, dimension, val):
        """
        Retrieve the nearest node ID.

        :param dimension: (str) "both", 'x', 'y' or 'z'
        :param val: (flt) Value of the dimension.
        :return: (int) ID of the node.
        """
        return np.argmin(np.abs(np.array(self.nodes_range(dimension)) - val))
