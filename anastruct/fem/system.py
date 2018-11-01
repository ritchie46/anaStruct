import math
import collections
import copy
import numpy as np
from anastruct.basic import FEMException
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.elements import Element
from anastruct.vertex import Vertex
from anastruct.fem import plotter
from . import system_components
from anastruct.vertex import vertex_range


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
        self.plotter = plotter.Plotter(self, mesh, plot_backend)

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

        # save tuples of the arguments for copying purposes.
        self.supports_spring_args = []

        # keep track of the loads
        self.loads_point = {}  # node ids with a point loads
        self.loads_q = {}  # element ids with a q-load
        self.loads_moment = {}
        self.loads_dead_load = []  # element ids with q-load due to dead load

        # results
        self.reaction_forces = {}  # node objects
        self.non_linear = False
        self.non_linear_elements = {}  # keys are element ids, values are dicts: {node_index: max moment capacity}
        self.buckling_factor = None

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
        """
        Add multiple elements defined by two containers with coordinates.

        :param x: (list/ np.array) x coordinates.
        :param y: (list/ np.array) y coordinates.
        :param EA: See 'add_element' method
        :param EI: See 'add_element' method
        :param g: See 'add_element' method
        :param mp: See 'add_element' method
        :param spring: See 'add_element' method
        :paramg **kwargs: See 'add_element' method
        :return: None
        """
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

        point_1, point_2 = system_components.util.det_vertices(self, location)
        node_id1, node_id2 = system_components.util.det_node_ids(self, point_1, point_2)

        point_1, point_2, node_id1, node_id2, spring, mp, ai = \
            system_components.util.force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp)

        system_components.util.append_node_id(self, point_1, point_2, node_id1, node_id2)
        system_components.util.ensure_single_hinge(self, spring, node_id1, node_id2)

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
            self.non_linear_elements[element.id] = mp
            self.non_linear = True
        system_components.assembly.dead_load(self, g, element.id)

        return self.count

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

        point_1, point_2 = system_components.util.det_vertices(self, location)
        length = (point_2 - point_1).modulus()
        direction = (point_2 - point_1).unit()

        if type(n) == type(dl):
            raise FEMException("Wrong parameters", "One, and only one, of n and dl should be passed as argument.")
        elif n:
            dl = length / n

        point = point_1 + direction * dl
        elements = [
            self.add_element((point_1, point), first["EA"], first["EI"], first["g"], first["mp"], first["spring"],
                             element_type=first["element_type"])]

        l = 2 * dl
        while l < length:
            point += direction * dl
            elements.append(self.add_element(point, EA, EI, g, mp, spring, element_type=element_type))
            l += dl

        elements.append(self.add_element(point_2, last["EA"], last["EI"], last["g"], last["mp"], last["spring"],
                                         element_type=last["element_type"]))
        return elements

    def solve(self, force_linear=False, verbosity=0, max_iter=200, geometrical_non_linear=False, **kwargs):

        """
        Compute the results of current model.

        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0. Log calculation outputs. 1. silence.
        :param max_iter: (int) Maximum allowed iterations.
        :param geometrical_non_linear: (bool) Calculate second order effects and determine the buckling factor.
        :return: (array) Displacements vector.


        Development **kwargs:
            :param naked: (bool) Whether or not to run the solve function without doing post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or increase the number
                                      of elements created that are used for determining the buckling_factor
        """

        # kwargs: arguments for the iterative solver callers such as the _stiffness_adaptation method.
        #                naked (bool) Default = False, if True force lines won't be computed.

        naked = kwargs.get("naked", False)

        if not naked:
            if not self.validate():
                if any(['general' in element.type for element in self.element_map.values()]):
                    raise FEMException('StabilityError', 'The eigenvalues of the stiffness matrix are non zero, '
                                                         'which indicates a instable structure. '
                                                         'Check your support conditions')

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        system_components.assembly.prep_matrix_forces(self)
        assert (self.system_force_vector is not None), "There are no forces on the structure"

        if self.non_linear and not force_linear:
            return system_components.solver.stiffness_adaptation(self, verbosity, max_iter)

        system_components.assembly.assemble_system_matrix(self)
        if geometrical_non_linear:
            discretize_kwargs = kwargs.get('discretize_kwargs', None)
            self.buckling_factor = system_components.solver.geometrically_non_linear(self, verbosity,
                                                                                     discretize_kwargs=discretize_kwargs,
                                                                                     discretize=discretize)
            return self.system_displacement_vector

        system_components.assembly.process_conditions(self)

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
            assert (np.any(self.system_displacement_vector < 1e6)), "The displacements of the structure exceed 1e6. " \
                                                                    "Check your support conditions," \
                                                                    "or your elements Young's modulus"

        return self.system_displacement_vector

    def validate(self, min_eigen=1e-9):
        """
        Validate the stability of the stiffness matrix.

        :param min_eigen: (flt) Minimum value of the eigenvalues of the stiffness matrix. This value should be close
        to zero.
        :return: (bool)
        """

        ss = copy.copy(self)
        system_components.assembly.prep_matrix_forces(ss)
        assert (np.abs(ss.system_force_vector).sum() != 0), "There are no forces on the structure"
        ss._remainder_indexes = []
        system_components.assembly.assemble_system_matrix(ss)

        system_components.assembly.process_conditions(ss)

        w, _ = np.linalg.eig(ss.reduced_system_matrix)
        return np.all(w > min_eigen)

    def add_support_hinged(self, node_id):
        """
        Model a hinged support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)
            system_components.assembly.set_displacement_vector(self, [(id_, 1), (id_, 2)])

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_roll(self, node_id, direction=2):
        """
        Adds a rolling support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        :param direction: (int/ list) Represents the direction that is fixed: x = 1, y = 2
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)
            system_components.assembly.set_displacement_vector(self, [(id_, direction)])

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction)

    def add_support_fixed(self, node_id):
        """
        Add a fixed support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)
            system_components.assembly.set_displacement_vector(self, [(id_, 1), (id_, 2), (id_, 3)])

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
        self.supports_spring_args.append((node_id, translation, k, roll))
        # The stiffness of the spring is added in the system matrix at the location that represents the node and the
        # displacement.
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

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
                    system_components.assembly.set_displacement_vector(self, [(id_, 2)])
                elif translation == 2:  # translation spring in z-axis
                    system_components.assembly.set_displacement_vector(self, [(id_, 1)])
                elif translation == 3:  # rotational spring in y-axis
                    system_components.assembly.set_displacement_vector(self, [(id_, 1), (id_, 2)])

    def q_load(self, q, element_id, direction="element"):
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"

        """
        if not isinstance(element_id, collections.Iterable):
            element_id = (element_id,)
        elif not hasattr(element_id, '__getitem__'):
            element_id = tuple(element_id)
        if not isinstance(q, collections.Iterable):
            q = (q,)
        elif not hasattr(q, '__getitem__'):
            q = tuple(q)

        if len(q) != len(element_id):  # arguments should be q, [id_1, [id_2]
            q = [q[0] for _ in element_id]

        for i in range(len(element_id)):
            id_ = _negative_index_to_id(element_id[i], self.element_map.keys())
            self.plotter.max_q = max(self.plotter.max_q, abs(q[i]))
            self.loads_q[id_] = q[i] * self.orientation_cs * self.load_factor
            el = self.element_map[id_]
            el.q_load = q[i] * self.orientation_cs * self.load_factor
            el.q_direction = direction

    def point_load(self, node_id, Fx=0, Fz=0):
        """
        Apply a point load to a node.

        :param node_id: (int) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fz: (flt/ list) Force in global x direction.
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)
            Fx = (Fx,)
            Fz = (Fz,)
        elif Fx == 0:
            Fx = [0 for _ in Fz]
        elif Fz == 0:
            Fz = [0 for _ in Fx]

        for i in range(len(node_id)):
            id_ = _negative_index_to_id(node_id[i], self.node_map.keys())
            self.plotter.max_system_point_load = max(self.plotter.max_system_point_load,
                                                     (Fx[i] ** 2 + Fz[i] ** 2) ** 0.5)
            self.loads_point[id_] = (Fx[i], Fz[i] * self.orientation_cs)

    def moment_load(self, node_id, Ty):
        """
        Apply a moment on a node.

        :param node_id: (int) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)
            Ty = (Ty,)

        for i in range(len(node_id)):
            id_ = _negative_index_to_id(node_id[i], self.node_map.keys())
            self.loads_moment[id_] = Ty[i]

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
                          linear=False, values_only=False):
        """
        Plot the displacement.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param linear: (bool) Dont evaluate the displacement values in between the elements
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        if values_only:
            p = plotter.PlottingValues(self, 10)
            return p.displacements(factor, linear)

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
            return [node.uz for node in self.node_map.values()]  # - * -  = +
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
                else (x.vertex.x, x.vertex.y) if dimension == "both"
                else None,
                self.node_map.values()))

    def nearest_node(self, dimension, val):
        """
        Retrieve the nearest node ID.

        :param dimension: (str) "both", 'x', 'y' or 'z'
        :param val: (flt) Value of the dimension.
        :return: (int) ID of the node.
        """
        if dimension == "both":
            match = list(map(lambda x: x[1],
                             filter(lambda x: x[0][0] == val[0] and x[0][1] == val[1],
                                    zip(self.nodes_range("both"), self.node_map.keys())
                                    )
                             )
                         )
            return match[0] if len(match) > 0 else None
        else:
            return np.argmin(np.abs(np.array(self.nodes_range(dimension)) - val))


def discretize(system, n=10):
    """
    Takes an already defined :class:`.SystemElements` object and increases the number of elements.

    :param system: (:class:`.SystemElements`)
    :param n: (int) Divide the elements into n sub-elements.
    :return: (:class:`.SystemElements`)
    """
    ss = SystemElements(EA=system.EA, EI=system.EI, load_factor=system.load_factor,
                        mesh=system.plotter.mesh, plot_backend=system.plotter.backend)

    for element in system.element_map.values():
        g = system.element_map[element.id].dead_load
        mp = system.non_linear_elements[element.id] if element.id in system.non_linear_elements else None

        for i, v in enumerate(vertex_range(element.vertex_1, element.vertex_2, n)):
            if i == 0:
                last_v = v
                continue

            loc = [last_v, v]
            ss.add_element(loc, EA=element.EA, EI=element.EI, g=g, mp=mp, spring=element.springs)
            last_v = v

    # supports
    for node, direction in zip(system.supports_roll, system.supports_roll_direction):
        ss.add_support_roll((node.id - 1) * n + 1, direction)
    for node in system.supports_fixed:
        ss.add_support_fixed((node.id - 1) * n + 1)
    for node in system.supports_hinged:
        ss.add_support_hinged((node.id - 1) * n + 1)
    for args in system.supports_spring_args:
        ss.add_support_spring((args[0] - 1) * n + 1, *args[1:])

    # loads
    for node_id, forces in system.loads_point.items():
        ss.point_load((node_id - 1) * n + 1, Fx=forces[0], Fz=forces[1] / system.orientation_cs)
    for node_id, forces in system.loads_moment.items():
        ss.moment_load((node_id - 1) * n + 1, forces)
    for element_id, forces in system.loads_q.items():
        ss.q_load(q=forces / system.orientation_cs / system.load_factor,
                  element_id=element_id,
                  direction=system.element_map[element_id].q_direction)

    return ss


def insert_node(system, location, element_id):
    ss = SystemElements(EA=system.EA, EI=system.EI, load_factor=system.load_factor,
                        mesh=system.plotter.mesh, plot_backend=system.plotter.backend)

    for element in system.element_map.values():
        g = system.element_map[element.id].dead_load
        mp = system.non_linear_elements[element.id] if element.id in system.non_linear_elements else None
        ss.add_element([element.vertex_1, element.vertex_2], EA=element.EA,
                       EI=element.EI, g=g, mp=mp, spring=element.springs)


def _negative_index_to_id(idx, collection):
    if idx > 0:
        return idx
    else:
        return max(collection) + (idx + 1)
