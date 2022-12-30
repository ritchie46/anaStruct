from __future__ import annotations
from math import sin, cos
from functools import lru_cache
import copy
from typing import TYPE_CHECKING, Dict, Optional, List
import numpy as np
from anastruct.basic import FEMException


if TYPE_CHECKING:
    from anastruct.vertex import Vertex
    from anastruct.fem.node import Node
    from anastruct.fem.system import Spring

try:
    from anastruct.fem.cython.celements import (  # type: ignore # pylint: disable=unused-import
        det_shear,
        det_moment,
        det_axial,
    )
except ImportError:
    from anastruct.fem.cython.elements import det_shear, det_moment, det_axial

"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""

CACHE_BOUND = 32000


class Element:
    def __init__(
        self,
        id_: int,
        EA: float,
        EI: float,
        l: float,
        angle: float,
        vertex_1: Vertex,
        vertex_2: Vertex,
        type_: str,
        section_name: str,
        spring: Spring = None,
    ):
        """
        :param id_: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param angle: angle between element and x-axis
        :param vertex_1: point object
        :param vertex_2: point object
        :param spring: (dict) Set a spring at node 1 or node 2.
                    spring={1: k
                            2: k}
        """
        self.id = id_
        self.type = type_
        self.EA = EA
        self.EI = EI
        self.l = l
        self.springs = spring
        self.vertex_1 = vertex_1  # location
        self.vertex_2 = vertex_2  # location
        self.angle = self.a1 = self.a2 = angle
        self.kinematic_matrix = kinematic_matrix(angle, angle, l)
        self.constitutive_matrix: np.ndarray
        self.stiffness_matrix: np.ndarray
        self.node_id1: int
        self.node_id2: int
        self.node_map: Dict[int, Node]
        self.element_displacement_vector: np.ndarray = np.empty(6)
        self.element_primary_force_vector: np.ndarray = np.zeros(
            6
        )  # acting external forces
        self.element_force_vector: np.ndarray = np.array([])
        self.q_load: tuple = (0.0, 0.0)
        self.q_perp_load: tuple = (0.0, 0.0)
        self.q_direction: Optional[str] = None
        self.q_angle: Optional[float] = None
        self.dead_load: float = 0.0
        self.N_1: Optional[float] = None
        self.N_2: Optional[float] = None
        self.bending_moment: Optional[np.ndarray] = None
        self.shear_force: Optional[np.ndarray] = None
        self.axial_force: Optional[np.ndarray] = None
        self.deflection: Optional[np.ndarray] = None
        self.extension: Optional[np.ndarray] = None
        self.max_deflection = None
        self.max_extension = None
        self.nodes_plastic: List[bool] = [False, False]
        self.compile_constitutive_matrix(EA, EI, l, spring)
        self.compile_stiffness_matrix()
        self.section_name = section_name  # needed for element annotation

    @property
    def all_qp_load(self) -> List[float]:
        if self.q_angle is not None:
            q_factor = sin(self.q_angle - self.angle)
            q_perp_factor = cos(self.q_angle - self.angle)
        else:
            q_factor = 1
            q_perp_factor = 0
        q = [
            i * q_factor + j * q_perp_factor
            for i, j in zip(self.q_load, self.q_perp_load)
        ]

        return [i + self.dead_load * cos(self.angle) for i in q]

    @property
    def all_qn_load(self) -> List[float]:
        if self.q_angle is not None:
            q_factor = -cos(self.q_angle - self.angle)
            q_perp_factor = sin(self.q_angle - self.angle)
        else:
            q_factor = 0
            q_perp_factor = 1
        q = [
            i * q_factor + j * q_perp_factor
            for i, j in zip(self.q_load, self.q_perp_load)
        ]

        return [i + self.dead_load * -sin(self.angle) for i in q]

    @property
    def node_1(self) -> Node:
        return self.node_map[self.node_id1]

    @property
    def node_2(self) -> Node:
        return self.node_map[self.node_id2]

    @property
    def hinges(self) -> List[int]:
        """
        Node ids of hinges on element
        """
        out = []

        assert self.springs is not None
        for k, v in self.springs.items():
            if v == 0:
                if k == 1:
                    out.append(self.node_id1)
                if k == 2:
                    out.append(self.node_id2)
        return out

    def determine_force_vector(self) -> Optional[np.ndarray]:
        self.element_force_vector = np.dot(
            self.stiffness_matrix, self.element_displacement_vector
        )
        return self.element_force_vector

    def compile_stiffness_matrix(self):
        self.stiffness_matrix = stiffness_matrix(
            self.constitutive_matrix, self.kinematic_matrix
        )

    def compile_kinematic_matrix(self, a1: float, a2: float, l: float):
        self.kinematic_matrix = kinematic_matrix(a1, a2, l)

    def compile_constitutive_matrix(
        self,
        EA: float,
        EI: float,
        l: float,
        spring: Optional[Dict[int, float]] = None,
        node_1_hinge: Optional[bool] = False,
        node_2_hinge: Optional[bool] = False,
    ):
        self.constitutive_matrix = constitutive_matrix(
            EA, EI, l, spring, node_1_hinge, node_2_hinge
        )

    def update_stiffness(self, factor: float, node: int):
        if node == 1:
            self.constitutive_matrix[1][1] *= factor
            self.constitutive_matrix[1][2] *= factor
            self.constitutive_matrix[2][1] *= factor
        elif node == 2:
            self.constitutive_matrix[1][2] *= factor
            self.constitutive_matrix[2][1] *= factor
            self.constitutive_matrix[2][2] *= factor
        self.compile_stiffness_matrix()

    def compile_geometric_non_linear_stiffness_matrix(self):
        self.compile_stiffness_matrix()
        self.stiffness_matrix += geometric_stiffness_matrix(
            self.l, self.N_1, self.a1, self.a2
        )

    def reset(self):
        self.element_displacement_vector = np.zeros(6)
        self.element_primary_force_vector = np.zeros(6)

    def __add__(self, other: Element) -> Element:
        if self.id != other.id:
            raise FEMException(
                "Wrong element:", "only elements with the same id can be added."
            )
        el = copy.deepcopy(self)
        for unit in [
            "bending_moment",
            "shear_force",
            "deflection",
            "extension",
            "N_1",
            "N_2",
            "axial_force",
        ]:
            if getattr(el, unit) is None:
                setattr(el, unit, getattr(other, unit))
            else:
                setattr(el, unit, getattr(el, unit) + getattr(other, unit))
        el.max_deflection = (
            other.max_deflection
            if el.max_deflection is None
            else max(el.max_deflection, other.max_deflection)
        )

        el.node_map[self.node_id1] = el.node_1 + other.node_1
        el.node_map[self.node_id2] = el.node_2 + other.node_2
        el.node_map[self.node_id1].ux = el.node_1.ux + other.node_1.ux
        el.node_map[self.node_id1].uz = el.node_1.uz + other.node_1.uz
        el.node_map[self.node_id1].phi_y = el.node_1.phi_y + other.node_1.phi_y
        el.node_map[self.node_id2].ux = el.node_2.ux + other.node_2.ux
        el.node_map[self.node_id2].uz = el.node_2.uz + other.node_2.uz
        el.node_map[self.node_id2].phi_y = el.node_2.phi_y + other.node_2.phi_y
        return el


@lru_cache(CACHE_BOUND)
def kinematic_matrix(a1: float, a2: float, l: float) -> np.ndarray:
    """
    Kinematic matrix of an element dependent of the angle ai and the length of the element.

    :param a1: (float) angle with respect to the x axis.
    :param l: (float) Length
    """
    c1 = cos(a1)
    s1 = sin(a1)
    c2 = cos(a2)
    s2 = sin(a2)
    return np.array(
        [
            [-c1, s1, 0, c2, -s2, 0],
            [s1 / l, c1 / l, -1, -s2 / l, -c2 / l, 0],
            [-s1 / l, -c1 / l, 0, s2 / l, c2 / l, 1],
        ]
    )


def constitutive_matrix(
    EA: float,
    EI: float,
    l: float,
    spring: Optional[Dict[int, float]],
    node_1_hinge: Optional[bool],
    node_2_hinge: Optional[bool],
) -> np.ndarray:
    """
    :param EA: (float) Young's modules * Area
    :param EI: (float) Young's modules * Moment of Inertia
    :param l: (float) Length
    :param spring: (int) 1 or 2. Apply a hinge on the first of the second node.
    :return: (array)
    """
    matrix = np.array(
        [[EA / l, 0, 0], [0, 4 * EI / l, -2 * EI / l], [0, -2 * EI / l, 4 * EI / l]]
    )

    if node_1_hinge or (spring is not None and 1 in spring and spring[1] == 0):
        matrix[1][1] = matrix[1][2] = matrix[2][1] = 0

    if node_2_hinge or (spring is not None and 2 in spring and spring[2] == 0):
        matrix[1][2] = matrix[2][1] = matrix[2][2] = 0

    if spring is not None:
        """
        stiffness matrix K:
        [[ k, k ]
        [ k, k ]]

        flexibility matrix C:
        [[ c, c ]    =   [[ 1/k, 1/k ]
        `[ c, c ]]       [  1/k, 1/k ]]

        flexibility matrix c + springs on both sides:
        [[ c11 + 1/k1, c12       ]
        [  c21      , c21 + 1/k2 ]]
        """
        if 1 in spring and spring[1] != 0:  # not hinge
            matrix[1][1] = 1 / (1 / matrix[1][1] + 1 / spring[1])
            matrix[2][1] = 1 / (1 / matrix[2][1] + 1 / spring[1])
        if 2 in spring and spring[2] != 0:  # not hinge
            matrix[2][1] = 1 / (1 / matrix[2][1] + 1 / spring[2])
            matrix[1][2] = 1 / (1 / matrix[1][2] + 1 / spring[2])

    return matrix


def stiffness_matrix(
    var_constitutive_matrix: np.ndarray, var_kinematic_matrix: np.ndarray
) -> np.ndarray:
    kinematic_transposed_times_constitutive = np.dot(
        var_kinematic_matrix.transpose(), var_constitutive_matrix
    )
    return np.dot(kinematic_transposed_times_constitutive, var_kinematic_matrix)


def geometric_stiffness_matrix(l: float, N: float, a1: float, a2: float) -> np.ndarray:
    """

    :param l: (float) Length.
    :param N: (float) Axial force.
    :param a1: (float) angle. (First try 1st order)
    :return: (array)
    """
    c1 = cos(a1)
    s1 = sin(a1)
    c2 = cos(a2)
    s2 = sin(a2)
    # http://people.duke.edu/~hpgavin/cee421/frame-finite-def.pdf
    return (
        N
        / l
        * np.array(
            [
                [
                    6 / 5 * s1**2,
                    -6 / 5 * s1 * c1,
                    -l / 10 * s1,
                    -6 / 5 * s2**2,
                    6 / 5 * s2 * c2,
                    -l / 10 * s2,
                ],
                [
                    -6 / 5 * s1 * c1,
                    6 / 5 * c1**2,
                    l / 10 * c1,
                    6 / 5 * s2 * c2,
                    -6 / 5 * c2**2,
                    l / 10 * c2,
                ],
                [
                    -l / 10 * s1,
                    l / 10 * c1,
                    2 * l**2 / 15,
                    l / 10 * s2,
                    -l / 10 * c2,
                    -(l**2) / 30,
                ],
                [
                    -6 / 5 * s1**2,
                    6 / 5 * s1 * c1,
                    l / 10 * s1,
                    6 / 5 * s2**2,
                    -6 / 5 * s1 * c2,
                    l / 10 * s2,
                ],
                [
                    6 / 5 * s1 * c1,
                    -6 / 5 * c1**2,
                    -l / 10 * c1,
                    -6 / 5 * s2 * c2,
                    6 / 5 * c2**2,
                    -l / 10 * c2,
                ],
                [
                    -l / 10 * s1,
                    l / 10 * c1,
                    -(l**2) / 30,
                    l / 10 * s2,
                    -l / 10 * c2,
                    2 * l**2 / 15,
                ],
            ]
        )
        * np.array([1, -1, 1, 1, -1, 1])
    )  # conversion from coordinate system
