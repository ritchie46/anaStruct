import math
import re
import collections.abc
import copy
from typing import (
    Tuple,
    Union,
    List,
    Optional,
    Dict,
    Set,
    TYPE_CHECKING,
    Sequence,
    cast,
    Collection,
    Any,
)
import numpy as np
from anastruct.basic import FEMException, args_to_lists
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.elements import Element
from anastruct.vertex import Vertex
from anastruct.fem import plotter
from anastruct.vertex import vertex_range
from anastruct.sectionbase import properties
from anastruct.fem.util.load import LoadCase
from . import system_components

if TYPE_CHECKING:
    from anastruct.fem.node import Node


Spring = Dict[int, float]
MpType = Dict[int, float]


class SystemElements:
    """
    Modelling any structure starts with an object of this class.

    :ivar EA: Standard axial stiffness of elements, default=15,000
    :ivar EI: Standard bending stiffness of elements, default=5,000
    :ivar figsize: (tpl) Matplotlibs standard figure size
    :ivar element_map: (dict) Keys are the element ids, values are the element objects
    :ivar node_map: (dict) Keys are the node ids, values are the node objects.
    :ivar node_element_map: (dict) maps node ids to element objects.
    :ivar loads_point: (dict) Maps node ids to point loads.
    :ivar loads_q: (dict) Maps element ids to q-loads.
    :ivar loads_moment: (dict) Maps node ids to moment loads.
    :ivar loads_dead_load: (set) Element ids that have a dead load applied.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        EA: float = 15e3,
        EI: float = 5e3,
        load_factor: float = 1.0,
        mesh: int = 50,
    ):
        """
        * E = Young's modulus
        * A = Area
        * I = Moment of Inertia

        :param figsize: Set the standard plotting size.
        :param EA: Standard E * A. Set the standard values of EA if none provided when
                   generating an element.
        :param EI: Standard E * I. Set the standard values of EA if none provided when
                   generating an element.
        :param load_factor: Multiply all loads with this factor.
        :param mesh: Plotting mesh. Has no influence on the calculation.
        """
        # init object
        self.post_processor = post_sl(self)
        self.plotter = plotter.Plotter(self, mesh)
        self.plot_values = plotter.PlottingValues(self, mesh)

        # standard values if none provided
        self.EA = EA
        self.EI = EI
        self.figsize = figsize
        self.orientation_cs = -1  # needed for the loads directions

        # structure system
        self.element_map: Dict[
            int, Element
        ] = {}  # maps element ids to the Element objects.
        self.node_map: Dict[
            int, Node  # pylint: disable=used-before-assignment
        ] = {}  # maps node ids to the Node objects.
        self.node_element_map: Dict[
            int, List[Element]
        ] = {}  # maps node ids to Element objects
        # keys matrix index (for both row and columns), values K, are processed
        # assemble_system_matrix
        self.system_spring_map: Dict[int, float] = {}

        # list of indexes that remain after conditions are applied
        self._remainder_indexes: List[int] = []

        # keep track of the nodes of the supports
        self.supports_fixed: List[Node] = []
        self.supports_hinged: List[Node] = []
        self.supports_rotational: List[Node] = []
        self.internal_hinges: List[Node] = []
        self.supports_roll: List[Node] = []
        self.supports_spring_x: List[Tuple[Node, bool]] = []
        self.supports_spring_z: List[Tuple[Node, bool]] = []
        self.supports_spring_y: List[Tuple[Node, bool]] = []
        self.supports_roll_direction: List[int] = []
        self.inclined_roll: Dict[
            int, float
        ] = {}  # map node ids to inclination angle relative to global x-axis.
        self.supports_roll_rotate: List[bool] = []

        # save tuples of the arguments for copying purposes.
        self.supports_spring_args: List[tuple] = []

        # keep track of the loads
        self.loads_point: Dict[
            int, Tuple[float, float]
        ] = {}  # node ids with a point loads {node_id: (x, y)}
        self.loads_q: Dict[
            int, List[Tuple[float, float]]
        ] = {}  # element ids with a q-loadad
        self.loads_moment: Dict[int, float] = {}
        self.loads_dead_load: Set[
            int
        ] = set()  # element ids with q-load due to dead load

        # results
        self.reaction_forces: Dict[int, Node] = {}  # node objects
        self.non_linear = False
        self.non_linear_elements: Dict[
            int, Dict[int, float]
        ] = (
            {}
        )  # keys are element ids, values are dicts: {node_index: max moment capacity}
        self.buckling_factor: Optional[float] = None

        # previous point of element
        self._previous_point = Vertex(0, 0)
        self.load_factor = load_factor

        # Objects state
        self.count = 0
        self.system_matrix: Optional[np.ndarray] = None
        self.system_force_vector: Optional[np.ndarray] = None
        self.system_displacement_vector: Optional[np.ndarray] = None
        self.shape_system_matrix: Optional[
            int
        ] = None  # actually is the size of the square system matrix
        self.reduced_force_vector: Optional[np.ndarray] = None
        self.reduced_system_matrix: Optional[np.ndarray] = None
        self._vertices: Dict[Vertex, int] = {}  # maps vertices to node ids

    @property
    def id_last_element(self) -> int:
        return max(self.element_map.keys())

    @property
    def id_last_node(self) -> int:
        return max(self.node_map.keys())

    def add_element_grid(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        EA: Optional[Union[List[float], np.ndarray]] = None,
        EI: Optional[Union[List[float], np.ndarray]] = None,
        g: Optional[Union[List[float], np.ndarray]] = None,
        mp: Optional[MpType] = None,
        spring: Optional[Spring] = None,
        **kwargs,
    ):
        """
        Add multiple elements defined by two containers with coordinates.

        :param x: x coordinates.
        :param y: y coordinates.
        :param EA: See 'add_element' method
        :param EI: See 'add_element' method
        :param g: See 'add_element' method
        :param mp: See 'add_element' method
        :param spring: See 'add_element' method
        :paramg **kwargs: See 'add_element' method
        :return: None
        """
        a = np.ones(len(x))
        if EA is None:
            EA = a * self.EA
        if EI is None:
            EI = a * self.EI
        if g is None:
            g = a * 0
        # cast to array if parameters are not given as array.
        EA_arr = EA * a
        EI_arr = EI * a
        g_arr = g * a

        for i in range(len(x) - 1):
            self.add_element(
                [[x[i], y[i]], [x[i + 1], y[i + 1]]],
                EA_arr[i],
                EI_arr[i],
                g_arr[i],
                mp,
                spring,
                **kwargs,
            )

    def add_truss_element(
        self,
        location: Union[
            Sequence[Sequence[float]], Sequence[Vertex], Sequence[float], Vertex
        ],
        EA: float = None,
        **kwargs,
    ) -> int:
        """
        .. highlight:: python

        Add an element that only has axial force.

        :param location: The two nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        :param EA: EA
        :return: Elements ID.
        """
        return self.add_element(location, EA, element_type="truss", **kwargs)

    def add_element(
        self,
        location: Union[
            Sequence[Sequence[float]], Sequence[Vertex], Sequence[float], Vertex
        ],
        EA: float = None,
        EI: float = None,
        g: float = 0,
        mp: Optional[MpType] = None,
        spring: Optional[Spring] = None,
        **kwargs,
    ) -> int:
        """
        :param location: The two nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        :param EA: EA
        :param EI: EI
        :param g: Weight per meter. [kN/m] / [N/m]
        :param mp: Set a maximum plastic moment capacity. Keys are integers representing
                   the nodes. Values are the bending moment capacity.

            :Example:

            .. code-block:: python

                mp={1: 210e3,
                    2: 180e3}

        :param spring: Set a rotational spring or a hinge (k=0) at node 1 or node 2.

            :Example:

            .. code-block:: python

                spring={1: k
                        2: k}


                # Set a hinged node:
                spring={1: 0}


        :return: Elements ID.
        """

        if mp is None:
            mp = {}
        if spring is None:
            spring = {}

        element_type = kwargs.get("element_type", "general")

        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        section_name = ""
        # change EA EI and g if steel section specified
        if "steelsection" in kwargs:
            section_name, EA, EI, g = properties.steel_section_properties(**kwargs)
        # change EA EI and g if rectangle section specified
        if "h" in kwargs:
            section_name, EA, EI, g = properties.rectangle_properties(**kwargs)
        # change EA EI and g if circle section specified
        if "d" in kwargs:
            section_name, EA, EI, g = properties.circle_properties(**kwargs)

        if element_type == "truss":
            EI = 1e-14

        # add the element number
        self.count += 1

        point_1, point_2 = system_components.util.det_vertices(self, location)
        node_id1, node_id2 = system_components.util.det_node_ids(self, point_1, point_2)

        (
            point_1,
            point_2,
            node_id1,
            node_id2,
            spring,
            mp,
            angle,
        ) = system_components.util.force_elements_orientation(
            point_1, point_2, node_id1, node_id2, spring, mp
        )

        system_components.util.append_node_id(
            self, point_1, point_2, node_id1, node_id2
        )

        # Only for typing purposes
        EA = cast(float, EA)
        EI = cast(float, EI)

        # add element
        element = Element(
            id_=self.count,
            EA=EA,
            EI=EI,
            l=(point_2 - point_1).modulus(),
            angle=angle,
            vertex_1=point_1,
            vertex_2=point_2,
            type_=element_type,
            spring=spring,
            section_name=section_name,
        )
        element.node_id1 = node_id1
        element.node_id2 = node_id2
        element.node_map = {
            node_id1: self.node_map[node_id1],
            node_id2: self.node_map[node_id2],
        }

        self.element_map[self.count] = element

        for node in (node_id1, node_id2):
            if node in self.node_element_map:
                self.node_element_map[node].append(element)
            else:
                self.node_element_map[node] = [element]

        # Register the elements per node
        for node_id in (node_id1, node_id2):
            self.node_map[node_id].elements[element.id] = element

        assert mp is not None
        if len(mp) > 0:
            assert isinstance(mp, dict), "The mp parameter should be a dictionary."
            self.non_linear_elements[element.id] = mp
            self.non_linear = True
        system_components.assembly.dead_load(self, g, element.id)

        return self.count

    def add_multiple_elements(
        self,
        location: Union[
            Sequence[Sequence[float]], Sequence[Vertex], Sequence[float], Vertex
        ],
        n: Optional[int] = None,
        dl: Optional[float] = None,
        EA: float = None,
        EI: float = None,
        g: float = 0,
        mp: Optional[MpType] = None,
        spring: Optional[Spring] = None,
        **kwargs,
    ):
        """
        Add multiple elements defined by the first and the last point.

        :param location: See 'add_element' method
        :param n: Number of elements.
        :param dl: Distance between the elements nodes.
        :param EA: See 'add_element' method
        :param EI: See 'add_element' method
        :param g: See 'add_element' method
        :param mp: See 'add_element' method
        :param spring: See 'add_element' method

        **Keyword Args:**

        :param element_type: See 'add_element' method
        :param first: Different arguments for the first element
        :param last: Different arguments for the last element
        :param steelsection: Steel section name like IPE 300
        :param orient: Steel section axis for moment of inertia - 'y' and 'z' possible
        :param b: Width of generic rectangle section
        :param h: Height of generic rectangle section
        :param d: Diameter of generic circle section
        :param sw: If true self weight of section is considered as dead load
        :param E: Modulus of elasticity for section material
        :param gamma: Weight of section material per volume unit. [kN/m3] / [N/m3]s

            :Example:

            .. code-block:: python

                last={'EA': 1e3, 'mp': 290}

        :return: (list) Element IDs
        """

        if mp is None:
            mp = {}
        if spring is None:
            spring = {}

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

        if n is not None and dl is not None:
            raise FEMException(
                "Wrong parameters",
                "One, and only one, of n and dl should be passed as argument.",
            )
        elif n:
            var_n = n
            lengths = np.linspace(start=0, stop=length, num=var_n + 1)
        else:
            assert dl is not None
            var_n = int(np.ceil(length / dl) - 1)
            lengths = np.linspace(start=0, stop=var_n * dl, num=var_n + 1)
            lengths = np.append(lengths, length)

        point = point_1 + direction * lengths[1]
        elements = [
            self.add_element(
                [point_1, point],
                first["EA"],
                first["EI"],
                first["g"],
                first["mp"],
                first["spring"],
                element_type=first["element_type"],
                **kwargs,
            )
        ]

        for i in range(2, var_n):
            point = point_1 + direction * lengths[i]
            elements.append(
                self.add_element(
                    point, EA, EI, g, mp, spring, element_type=element_type, **kwargs
                )
            )

        elements.append(
            self.add_element(
                point_2,
                last["EA"],
                last["EI"],
                last["g"],
                last["mp"],
                last["spring"],
                element_type=last["element_type"],
                **kwargs,
            )
        )
        return elements

    def insert_node(
        self,
        element_id: int,
        location: Union[Sequence[float], Vertex, None] = None,
        factor=None,
    ):
        """
        Insert a node into an existing structure.
        This can be done by adding a new Vertex at any given location, or by setting
        a factor of the elements length. E.g. if you want a node at 40% of the elements
        length, you pass factor = 0.4.

        Note: this method completely rebuilds the SystemElements object and is therefore
        slower then building a model with `add_element` methods.

        :param element_id: Id number of the element you want to insert the node.
        :param location: The nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[x, y]
                   location=Vertex

        :param: factor: Value between 0 and 1 to determine the new node location.
        """
        ss = SystemElements(
            EA=self.EA, EI=self.EI, load_factor=self.load_factor, mesh=self.plotter.mesh
        )

        for element in self.element_map.values():
            g = self.element_map[element.id].dead_load
            mp = (
                self.non_linear_elements[element.id]
                if element.id in self.non_linear_elements
                else {}
            )
            if element_id == element.id:
                if factor is not None:
                    location_vertex = Vertex(
                        factor * (element.vertex_2 - element.vertex_1)
                        + element.vertex_1
                    )
                else:
                    assert location is not None
                    location_vertex = Vertex(location)

                mp1 = mp2 = spring1 = spring2 = {}
                if len(mp) != 0:
                    if 1 in mp:
                        mp1 = {1: mp[1]}
                    if 2 in mp:
                        mp2 = {2: mp[2]}
                if element.springs is not None:
                    if 1 in element.springs:
                        spring1 = {1: element.springs[1]}
                    if 2 in element.springs:
                        spring2 = {2: element.springs[2]}

                ss.add_element(
                    [element.vertex_1, location_vertex],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp1,
                    spring=spring1,
                )
                ss.add_element(
                    [location_vertex, element.vertex_2],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp2,
                    spring=spring2,
                )

            else:
                ss.add_element(
                    [element.vertex_1, element.vertex_2],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp,
                    spring=element.springs,
                )
        self.__dict__ = ss.__dict__.copy()

    def solve(
        self,
        force_linear: bool = False,
        verbosity: int = 0,
        max_iter: int = 200,
        geometrical_non_linear: int = False,
        **kwargs,
    ):

        """
        Compute the results of current model.

        :param force_linear: Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: 0. Log calculation outputs. 1. silence.
        :param max_iter: Maximum allowed iterations.
        :param geometrical_non_linear: Calculate second order effects and determine the
                                       buckling factor.
        :return: Displacements vector.


        Development **kwargs:
            :param naked: Whether or not to run the solve function without doing post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or
                                      increase the number of elements created that are used for
                                      determining the buckling_factor
        """

        # kwargs: arguments for the iterative solver callers such as the _stiffness_adaptation
        #         method.
        #                naked (bool) Default = False, if True force lines won't be computed.

        for node_id in self.node_map:
            system_components.util.check_internal_hinges(self, node_id)

        if self.system_displacement_vector is None:
            system_components.assembly.process_supports(self)

        naked = kwargs.get("naked", False)

        if not naked:
            if not self.validate():
                if all(
                    ["general" in element.type for element in self.element_map.values()]
                ):
                    raise FEMException(
                        "StabilityError",
                        "The eigenvalues of the stiffness matrix are non zero, "
                        "which indicates a instable structure. "
                        "Check your support conditions",
                    )

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        system_components.assembly.prep_matrix_forces(self)
        assert (
            self.system_force_vector is not None
        ), "There are no forces on the structure"

        if self.non_linear and not force_linear:
            return system_components.solver.stiffness_adaptation(
                self, verbosity, max_iter
            )

        system_components.assembly.assemble_system_matrix(self)
        if geometrical_non_linear:
            discretize_kwargs = kwargs.get("discretize_kwargs", None)
            self.buckling_factor = system_components.solver.geometrically_non_linear(
                self,
                verbosity,
                return_buckling_factor=True,
                discretize_kwargs=discretize_kwargs,
            )
            return self.system_displacement_vector

        system_components.assembly.process_conditions(self)

        # solution of the reduced system (reduced due to support conditions)
        reduced_displacement_vector = np.linalg.solve(
            self.reduced_system_matrix, self.reduced_force_vector
        )

        # add the solution of the reduced system in the complete system displacement vector
        assert self.shape_system_matrix is not None
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        np.put(
            self.system_displacement_vector,
            self._remainder_indexes,
            reduced_displacement_vector,
        )

        # determine the displacement vector of the elements
        for el in self.element_map.values():
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

            # node 1 ux, uz, phi
            el.element_displacement_vector[:3] = self.system_displacement_vector[
                index_node_1 : index_node_1 + 3
            ]
            # node 2 ux, uz, phi
            el.element_displacement_vector[3:] = self.system_displacement_vector[
                index_node_2 : index_node_2 + 3
            ]
            el.determine_force_vector()

        if not naked:
            # determining the node results in post processing class
            self.post_processor.node_results_elements()
            self.post_processor.node_results_system()
            self.post_processor.reaction_forces()
            self.post_processor.element_results()

            # check the values in the displacement vector for extreme values, indicating a
            # flawed calculation
            assert np.any(self.system_displacement_vector < 1e6), (
                "The displacements of the structure exceed 1e6. "
                "Check your support conditions,"
                "or your elements Young's modulus"
            )

        return self.system_displacement_vector

    def validate(self, min_eigen: float = 1e-9) -> bool:
        """
        Validate the stability of the stiffness matrix.

        :param min_eigen: Minimum value of the eigenvalues of the stiffness matrix. This value
        should be close to zero.
        """

        ss = copy.copy(self)
        system_components.assembly.prep_matrix_forces(ss)
        assert (
            np.abs(ss.system_force_vector).sum() != 0
        ), "There are no forces on the structure"
        ss._remainder_indexes = []
        system_components.assembly.assemble_system_matrix(ss)

        system_components.assembly.process_conditions(ss)

        w, _ = np.linalg.eig(ss.reduced_system_matrix)
        return bool(np.all(w > min_eigen))

    def add_support_hinged(self, node_id: Union[int, Sequence[int]]):
        """
        Model a hinged support at a given node.

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_rotational(self, node_id: Union[int, Sequence[int]]):
        """
        Model a rotational support at a given node.

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.supports_rotational.append(self.node_map[id_])

    def add_internal_hinge(self, node_id: Union[int, Sequence[int]]):
        """
        Model an internal hinge at a given node.
        This may alternately be done by setting spring={n: 0} when creating elements
        but this can be an easier method of doing so

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.internal_hinges.append(self.node_map[id_])

    def add_support_roll(
        self,
        node_id: Union[Sequence[int], int],
        direction: Union[Sequence[Union[str, int]], Union[str, int]] = "x",
        angle: Union[Sequence[Optional[float]], Optional[float]] = None,
        rotate: Union[Sequence[bool], bool] = True,
    ):
        """
        Adds a rolling support at a given node.

        :param node_id: Represents the nodes ID
        :param direction: Represents the direction that is free: 'x', 'y'
        :param angle: Angle in degrees relative to global x-axis.
                                If angle is given, the support will be inclined.
        :param rotate: If set to False, rotation at the roller will also be restrained.
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]
        if not isinstance(direction, collections.abc.Iterable):
            direction = [direction]
        if not isinstance(angle, collections.abc.Iterable):
            angle = [angle]
        if not isinstance(rotate, collections.abc.Iterable):
            rotate = [rotate]

        assert len(node_id) == len(direction) == len(angle) == len(rotate)

        for id_, direction_, angle_, rotate_ in zip(node_id, direction, angle, rotate):
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            if direction_ == "x":
                direction_i = 2
            elif direction_ == "y":
                direction_i = 1
            else:
                direction_i = int(direction_)

            if angle_ is not None:
                direction_i = 2
                self.inclined_roll[id_] = float(np.radians(-angle_))

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction_i)
            self.supports_roll_rotate.append(rotate_)

    def add_support_fixed(
        self,
        node_id: Union[Sequence[int], int],
    ):
        """
        Add a fixed support at a given node.

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [
                node_id,
            ]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def add_support_spring(
        self,
        node_id: Union[Sequence[int], int],
        translation: Union[Sequence[int], int],
        k: Union[Sequence[float], float],
        roll: Union[Sequence[bool], bool] = False,
    ):
        """
        Add a translational support at a given node.

        :param translation: Represents the prevented translation.

            **Note**

            |  1 = translation in x
            |  2 = translation in z
            |  3 = rotation in y

        :param node_id: Integer representing the nodes ID.
        :param k: Stiffness of the spring
        :param roll: If set to True, only the translation of the spring is controlled.

        """
        self.supports_spring_args.append((node_id, translation, k, roll))
        # The stiffness of the spring is added in the system matrix at the location that
        # represents the node and the displacement.
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = (node_id,)
        if not isinstance(translation, collections.abc.Iterable):
            translation = (translation,)
        if not isinstance(k, collections.abc.Iterable):
            k = (k,)
        if not isinstance(roll, collections.abc.Iterable):
            roll = (roll,)

        assert len(node_id) == len(translation) == len(k) == len(roll)

        for id_, translation_, k_, roll_ in zip(node_id, translation, k, roll):
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # determine the location in the system matrix
            # row and column are the same
            matrix_index = (id_ - 1) * 3 + translation_ - 1

            self.system_spring_map[matrix_index] = k_

            # add the support to the support list for the plotter
            if translation_ == 1:
                self.supports_spring_x.append((self.node_map[id_], roll_))
            elif translation_ == 2:
                self.supports_spring_z.append((self.node_map[id_], roll_))
            else:
                self.supports_spring_y.append((self.node_map[id_], roll_))

    def q_load(
        self,
        q: Union[float, Sequence[float]],
        element_id: Union[int, Sequence[int]],
        direction: Union[str, Sequence[str]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Union[float, Sequence[float]] = None,
    ):
        """
        Apply a q-load to an element.

        :param element_id: representing the element ID
        :param q: value of the q-load
        :param direction: "element", "x", "y", "parallel"
        :param rotation: Rotate the force clockwise. Rotation is in degrees
        :param q_perp: value of any q-load perpendicular to the indication direction/rotation
        """
        # TODO! this function is a duck typing hell. redesign.
        if q_perp is None:
            q_perp = [0, 0]
        if not isinstance(q, Sequence):
            q = [q, q]
        if not isinstance(q_perp, Sequence):
            q_perp = [q_perp, q_perp]
        q = [q]  # type: ignore
        q_perp = [q_perp]  # type: ignore
        if rotation is None:
            direction_flag = True
        else:
            direction_flag = False
        (  # pylint: disable=unbalanced-tuple-unpacking
            q,
            element_id,
            direction,
            rotation,
            q_perp,
        ) = args_to_lists(q, element_id, direction, rotation, q_perp)

        assert len(q) == len(element_id)  # type: ignore
        assert len(q) == len(direction)  # type: ignore
        assert len(q) == len(rotation)  # type: ignore
        assert len(q) == len(q_perp)  # type: ignore

        for i, element_idi in enumerate(element_id):  # type: ignore
            id_ = _negative_index_to_id(element_idi, self.element_map.keys())  # type: ignore
            self.plotter.max_q = max(
                self.plotter.max_q,
                max(
                    (q[i][0] ** 2 + q_perp[i][0] ** 2) ** 0.5,  # type: ignore
                    (q[i][1] ** 2 + q_perp[i][1] ** 2) ** 0.5,  # type: ignore
                ),
            )

            if direction_flag:
                if direction[i] == "x":
                    rotation[i] = 0  # type: ignore
                elif direction[i] == "y":
                    rotation[i] = np.pi / 2  # type: ignore
                elif direction[i] == "parallel":
                    rotation[i] = self.element_map[element_id[i]].angle  # type: ignore
                else:
                    rotation[i] = np.pi / 2 + self.element_map[element_id[i]].angle  # type: ignore
            else:
                rotation[i] = math.radians(rotation[i])  # type: ignore
                direction[i] = "angle"  # type: ignore

            cos = math.cos(rotation[i])  # type: ignore
            sin = math.sin(rotation[i])  # type: ignore
            self.loads_q[id_] = [
                (
                    (q_perp[i][0] * cos + q[i][0] * sin) * self.load_factor,  # type: ignore
                    (q[i][0] * self.orientation_cs * cos + q_perp[i][0] * sin)  # type: ignore
                    * self.load_factor,
                ),
                (
                    (q_perp[i][1] * cos + q[i][1] * sin) * self.load_factor,  # type: ignore
                    (q[i][1] * self.orientation_cs * cos + q_perp[i][1] * sin)  # type: ignore
                    * self.load_factor,
                ),
            ]
            el = self.element_map[id_]
            el.q_load = [i * self.orientation_cs * self.load_factor for i in q[i]]  # type: ignore
            el.q_perp_load = [i * self.load_factor for i in q_perp[i]]  # type: ignore
            el.q_direction = direction[i]  # type: ignore
            el.q_angle = rotation[i]  # type: ignore

    def point_load(
        self,
        node_id: Union[int, Sequence[int]],
        Fx: Union[float, Sequence[float]] = 0.0,
        Fy: Union[float, Sequence[float]] = 0.0,
        rotation: Union[float, Sequence[float]] = 0,
    ):
        """
        Apply a point load to a node.

        :param node_id: Nodes ID.
        :param Fx: Force in global x direction.
        :param Fy: Force in global x direction.
        :param rotation: Rotate the force clockwise. Rotation is in degrees.
        """
        (  # pylint: disable=unbalanced-tuple-unpacking
            node_id,
            Fx,
            Fy,
            rotation,
        ) = args_to_lists(node_id, Fx, Fy, rotation)
        node_id = cast(Sequence[int], node_id)
        Fx = cast(Sequence[float], Fx)
        Fy = cast(Sequence[float], Fy)
        rotation = cast(Sequence[float], rotation)

        assert len(node_id) == len(Fx) == len(Fy) == len(rotation)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            if (
                id_ in self.inclined_roll
                and np.mod(self.inclined_roll[id_], np.pi / 2) != 0
            ):
                raise FEMException(
                    "StabilityError",
                    "Point loads may not be placed at the location of "
                    "inclined roller supports",
                )
            self.plotter.max_system_point_load = max(
                self.plotter.max_system_point_load, (Fx[i] ** 2 + Fy[i] ** 2) ** 0.5
            )
            cos = math.cos(math.radians(rotation[i]))
            sin = math.sin(math.radians(rotation[i]))
            self.loads_point[id_] = (
                (Fx[i] * cos + Fy[i] * sin) * self.load_factor,
                (Fy[i] * self.orientation_cs * cos + Fx[i] * sin) * self.load_factor,
            )

    def moment_load(
        self, node_id: Union[int, Sequence[int]], Ty: Union[float, Sequence[float]]
    ):
        """
        Apply a moment on a node.

        :param node_id: Nodes ID.
        :param Ty: Moments acting on the node.
        """
        node_id, Ty = args_to_lists(  # pylint: disable=unbalanced-tuple-unpacking
            node_id, Ty
        )
        node_id = cast(Sequence[int], node_id)
        Ty = cast(Sequence[float], Ty)

        assert len(node_id) == len(Ty)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            self.loads_moment[id_] = Ty[i] * self.load_factor

    def show_structure(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        supports: bool = True,
        values_only: bool = False,
        annotations: bool = False,
    ):
        """
        Plot the structure.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        :param annotations: if True, structure annotations are plotted. It includes section name.
                                      Note: only works when verbosity is equal to 0.
        """
        figsize = self.figsize if figsize is None else figsize
        if values_only:
            return self.plot_values.structure()
        return self.plotter.plot_structure(
            figsize, verbosity, show, supports, scale, offset, annotations=annotations
        )

    def show_bending_moment(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Tuple[float, float] = None,
        show: bool = True,
        values_only: bool = False,
    ):
        """
        Plot the bending moment.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.bending_moment(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.bending_moment(
            factor, figsize, verbosity, scale, offset, show
        )

    def show_axial_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ):
        """
        Plot the axial force.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.axial_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    def show_shear_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ):
        """
        Plot the shear force.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.shear_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.shear_force(factor, figsize, verbosity, scale, offset, show)

    def show_reaction_force(
        self,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ):
        """
        Plot the reaction force.

        :param verbosity: 0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        linear: bool = False,
        values_only: bool = False,
    ):
        """
        Plot the displacement.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param linear: Don't evaluate the displacement values in between the elements
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.displacements(factor, linear)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.displacements(
            factor, figsize, verbosity, scale, offset, show, linear
        )

    def show_results(
        self,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ):
        """
        Plot all the results in one window.

        :param verbosity: 0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.results_plot(figsize, verbosity, scale, offset, show)

    def get_node_results_system(
        self, node_id: int = 0
    ) -> Union[
        List[Tuple[Any, Any, Any, Any, Any, Any, Any]], Dict[str, Union[int, float]]
    ]:
        """
        These are the node results. These are the opposite of the forces and displacements
        working on the elements and may seem counter intuitive.

        :param node_id:  representing the node's ID. If integer = 0, the results of all nodes
                         are returned
        :return:

        |  if node_id == 0:
        |
        |    Returns a list containing tuples with the results:

        ::

            [(id, Fx, Fy, Ty, ux, uy, phi_y), (id, Fx, Fy...), () .. ]
        |
        |  if node_id > 0:
        """
        result_list = []
        if node_id != 0:
            node_id = _negative_index_to_id(node_id, self.node_map)
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "Fx": node.Fx,
                "Fy": node.Fy,
                "Ty": node.Ty,
                "ux": node.ux,
                "uy": -node.uz,
                "phi_y": node.phi_y,
            }
        else:
            for node in self.node_map.values():
                result_list.append(
                    (node.id, node.Fx, node.Fy, node.Ty, node.ux, -node.uz, node.phi_y)
                )
        return result_list

    def get_node_displacements(
        self, node_id: int = 0
    ) -> Union[List[Tuple[Any, Any, Any, Any]], Dict[str, Any]]:
        """
        :param node_id: Represents the node's ID. If integer = 0, the results of all nodes
                        are returned.
        :return:
        |  if node_id == 0:
        |
        |    Returns a list containing tuples with the results:

        ::

             [(id, ux, uy, phi_y), (id, ux, uy, phi_y),  ... (id, ux, uy, phi_y) ]
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
                "phi_y": node.phi_y,
            }
        else:
            for node in self.node_map.values():
                result_list.append((node.id, -node.ux, node.uz, node.phi_y))
        return result_list

    def get_element_results(
        self, element_id: int = 0, verbose: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        :param element_id: representing the elements ID. If elementID = 0 the results
                           of all elements are returned.
        :param verbose: If set to True the numerical results for the deflection and the
                        bending moments are returned.

        :return:
        |
        |  if node_id == 0:
        |
        |    Returns a list containing tuples with the results:

        ::

             [(id, length, alpha, u, N_1, N_2), (id, length, alpha, u, N_1, N_2),
             ... (id, length, alpha, u, N_1, N_2)]
        |
        |  if node_id > 0: (dict)
        |

        """
        if element_id != 0:
            element_id = _negative_index_to_id(element_id, self.element_map)
            el = self.element_map[element_id]

            assert el.extension is not None
            assert el.axial_force is not None

            if el.type == "truss":
                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.angle,
                    "umax": np.max(el.extension),
                    "umin": np.min(el.extension),
                    "u": el.extension if verbose else None,
                    "Nmin": np.min(el.axial_force),
                    "Nmax": np.max(el.axial_force),
                    "N": el.axial_force if verbose else None,
                }
            else:
                assert el.deflection is not None
                assert el.shear_force is not None
                assert el.bending_moment is not None

                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.angle,
                    "umax": np.max(el.extension),
                    "umin": np.min(el.extension),
                    "u": el.extension if verbose else None,
                    "wmax": np.min(el.deflection),
                    "wmin": np.max(el.deflection),
                    "w": el.deflection if verbose else None,
                    "Mmin": np.min(el.bending_moment),
                    "Mmax": np.max(el.bending_moment),
                    "M": el.bending_moment if verbose else None,
                    "Qmin": np.min(el.shear_force),
                    "Qmax": np.max(el.shear_force),
                    "Q": el.shear_force if verbose else None,
                    "Nmin": np.min(el.axial_force),
                    "Nmax": np.max(el.axial_force),
                    "N": el.axial_force if verbose else None,
                    "q": el.q_load,
                }
        else:
            result_list = []
            for el in self.element_map.values():

                assert el.extension is not None
                assert el.axial_force is not None

                if el.type == "truss":
                    result_list.append(
                        {
                            "id": el.id,
                            "length": el.l,
                            "alpha": el.angle,
                            "umax": np.max(el.extension),
                            "umin": np.min(el.extension),
                            "u": el.extension if verbose else None,
                            "Nmin": np.min(el.axial_force),
                            "Nmax": np.max(el.axial_force),
                            "N": el.axial_force if verbose else None,
                        }
                    )

                else:
                    assert el.deflection is not None
                    assert el.shear_force is not None
                    assert el.bending_moment is not None

                    result_list.append(
                        {
                            "id": el.id,
                            "length": el.l,
                            "alpha": el.angle,
                            "umax": np.max(el.extension),
                            "umin": np.min(el.extension),
                            "u": el.extension if verbose else None,
                            "wmax": np.min(el.deflection),
                            "wmin": np.max(el.deflection),
                            "w": el.deflection if verbose else None,
                            "Mmin": np.min(el.bending_moment),
                            "Mmax": np.max(el.bending_moment),
                            "M": el.bending_moment if verbose else None,
                            "Qmin": np.min(el.shear_force),
                            "Qmax": np.max(el.shear_force),
                            "Q": el.shear_force if verbose else None,
                            "Nmin": np.min(el.axial_force),
                            "Nmax": np.max(el.axial_force),
                            "N": el.axial_force if verbose else None,
                            "q": el.q_load,
                        }
                    )
            return result_list

    def get_element_result_range(self, unit: str) -> List[float]:
        """
        Useful when added lots of elements. Returns a list of all the queried unit.

        :param unit:
            - 'shear'
            - 'moment'
            - 'axial'

        """
        if unit == "shear":
            return [el.shear_force[0] for el in self.element_map.values()]  # type: ignore
        elif unit == "moment":
            return [el.bending_moment[0] for el in self.element_map.values()]  # type: ignore
        elif unit == "axial":
            return [el.axial_force[0] for el in self.element_map.values()]  # type: ignore
        else:
            raise NotImplementedError

    def get_node_result_range(self, unit: str) -> List[float]:
        """
        Query a list with node results.

           :param unit:
            - 'uy'
            - 'ux'
            - 'phi_y'
        """
        if unit == "uy":
            return [node.uz for node in self.node_map.values()]  # - * -  = +
        elif unit == "ux":
            return [-node.ux for node in self.node_map.values()]
        elif unit == "phi_y":
            return [node.phi_y for node in self.node_map.values()]
        else:
            raise NotImplementedError

    def find_node_id(self, vertex: Union[Vertex, Sequence[float]]) -> Optional[int]:
        """
        Retrieve the ID of a certain location.

        :param vertex:  Vertex_xz, [x, y], (x, y)
        :return:  id of the node at the location of the vertex
        """
        if isinstance(vertex, (list, tuple)):
            vertex = Vertex(vertex)
        try:
            tol = 1e-9
            return next(  # type: ignore
                filter(
                    lambda x: math.isclose(x.vertex.x, vertex.x, abs_tol=tol)  # type: ignore
                    and math.isclose(x.vertex.y, vertex.y, abs_tol=tol),  # type: ignore
                    self.node_map.values(),
                )
            ).id
        except StopIteration:
            return None

    def nodes_range(
        self, dimension: str
    ) -> List[Union[float, Tuple[float, float], None]]:
        """
        Retrieve a list with coordinates x or z (y).

        :param dimension: "both", 'x', 'y' or 'z'
        """
        return list(
            map(
                lambda x: x.vertex.x
                if dimension == "x"
                else x.vertex.z
                if dimension == "z"
                else x.vertex.y
                if dimension == "y"
                else (x.vertex.x, x.vertex.y)
                if dimension == "both"
                else None,
                self.node_map.values(),
            )
        )

    def nearest_node(
        self, dimension: str, val: Union[float, Sequence[float]]
    ) -> Union[int, None]:
        """
        Retrieve the nearest node ID.

        :param dimension: "both", 'x', 'y' or 'z'
        :param val: Value of the dimension.
        :return:  ID of the node.
        """
        if dimension == "both" and isinstance(val, Sequence):
            match = list(
                map(
                    lambda x: x[1],  # type: ignore
                    filter(
                        lambda x: x[0][0] == val[0] and x[0][1] == val[1],  # type: ignore
                        zip(self.nodes_range("both"), self.node_map.keys()),
                    ),
                )
            )
            return match[0] if len(match) > 0 else None
        else:
            return int(np.argmin(np.abs(np.array(self.nodes_range(dimension)) - val)))

    def discretize(self, n: int = 10):
        """
        Takes an already defined :class:`.SystemElements` object and increases the number
        of elements.

        :param n: Divide the elements into n sub-elements.
        """
        ss = SystemElements(
            EA=self.EA, EI=self.EI, load_factor=self.load_factor, mesh=self.plotter.mesh
        )

        for element in self.element_map.values():
            g = self.element_map[element.id].dead_load
            mp = (
                self.non_linear_elements[element.id]
                if element.id in self.non_linear_elements
                else {}
            )

            for i, v in enumerate(vertex_range(element.vertex_1, element.vertex_2, n)):
                if i == 0:
                    last_v = v
                    continue

                loc = [last_v, v]
                ss.add_element(
                    loc,
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp,
                    spring=element.springs,
                )
                last_v = v

        # supports
        for node, direction, rotate in zip(
            self.supports_roll, self.supports_roll_direction, self.supports_roll_rotate
        ):
            ss.add_support_roll((node.id - 1) * n + 1, direction, None, rotate)
        for node in self.supports_fixed:
            ss.add_support_fixed((node.id - 1) * n + 1)
        for node in self.supports_hinged:
            ss.add_support_hinged((node.id - 1) * n + 1)
        for node in self.supports_rotational:
            ss.add_support_rotational((node.id - 1) * n + 1)
        for args in self.supports_spring_args:
            ss.add_support_spring((args[0] - 1) * n + 1, *args[1:])

        # loads
        for node_id, forces in self.loads_point.items():
            ss.point_load(
                (node_id - 1) * n + 1,
                Fx=forces[0] / self.load_factor,
                Fy=forces[1] / self.orientation_cs / self.load_factor,
            )

        for node_id, forces_moment in self.loads_moment.items():
            ss.moment_load((node_id - 1) * n + 1, forces_moment / self.load_factor)
        for element_id, forces_q in self.loads_q.items():
            ss.q_load(
                q=[i[1] / self.orientation_cs / self.load_factor for i in forces_q],
                element_id=element_id,
                direction="y",
                q_perp=[i[0] / self.load_factor for i in forces_q],
            )

        self.__dict__ = ss.__dict__.copy()

    def remove_loads(self, dead_load: bool = False):
        """
        Remove all the applied loads from the structure.

        :param dead_load: Remove the dead load.
        """

        self.loads_point = {}
        self.loads_q = {}
        self.loads_moment = {}

        for k in self.element_map:  # pylint: disable=consider-using-dict-items
            self.element_map[k].q_load = (0.0, 0.0)
            if dead_load:
                self.element_map[k].dead_load = 0
        if dead_load:
            self.loads_dead_load = set()

    def apply_load_case(self, loadcase: LoadCase):
        for method, kwargs in loadcase.spec.items():
            method = method.split("-")[0]
            kwargs = re.sub(r"[{}]", "", str(kwargs))
            # pass the groups that match back to the replace
            kwargs = re.sub(r".??(\w+).?:", r"\1=", kwargs)

            exec(f"self.{method}({kwargs})")  # pylint: disable=exec-used

    def __deepcopy__(self, memo):
        system = copy.copy(self)
        mesh = self.plotter.mesh
        system.plotter = None
        system.post_processor = None
        system.plot_values = None

        system.__dict__ = copy.deepcopy(system.__dict__)
        system.plotter = plotter.Plotter(system, mesh)
        system.post_processor = post_sl(system)
        system.plot_values = plotter.PlottingValues

        return system


def _negative_index_to_id(idx: int, collection: Collection[int]) -> int:
    if idx > 0:
        return idx
    else:
        return max(collection) + (idx + 1)
