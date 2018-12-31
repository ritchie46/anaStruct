from math import sin, cos
from anastruct.basic import FEMException
import numpy as np
from functools import lru_cache
import copy

try:
    from anastruct.fem.cython.celements import det_shear, det_moment
except ImportError:
    from anastruct.fem.cython.elements import det_shear, det_moment

"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""

CACHE_BOUND = 32000


class Element:
    def __init__(self, id_, EA, EI, l, angle, vertex_1, vertex_2, spring=None):
        """
        :param id_: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param angle: angle between element and x-axis
        :param vertex_1: point object
        :param vertex_2: point object
        :param hinge: (integer) 1 or 2. Adds an hinge ad the first or second node.
        :param spring: (dict) Set a spring at node 1 or node 2.
                            {
                              1: { direction: 1
                                   k: 5e3
                                 }
                              2: { direction: 1
                                   k: 5e3
                                 }
                            }
        """
        self.id = id_
        self.type = None
        self.EA = EA
        self.EI = EI
        self.l = l
        self.springs = spring
        self.vertex_1 = vertex_1  # location
        self.vertex_2 = vertex_2  # location
        self.angle = self.a1 = self.a2 = angle
        self.kinematic_matrix = kinematic_matrix(angle, angle, l)
        self.constitutive_matrix = None
        self.stiffness_matrix = None
        self.node_id1 = None  # int
        self.node_id2 = None  # int
        self.node_ids = []
        self.node_map = None
        self.element_displacement_vector = np.empty(6)
        self.element_primary_force_vector = np.zeros(6)  # acting external forces
        self.element_force_vector = None
        self.q_load = 0
        self.q_direction = None
        self.dead_load = 0
        self.N_1 = None
        self.N_2 = None
        self.bending_moment = None
        self.shear_force = None
        self.deflection = None
        self.extension = None
        self.max_deflection = None
        self.nodes_plastic = [False, False]
        self.compile_constitutive_matrix(self.EA, self.EI, l)
        self.compile_stiffness_matrix()

    @property
    def all_q_load(self):
        if self.q_load is None:
            q = 0
        else:
            if self.q_direction == "x":
                q_factor = -sin(self.angle)
            elif self.q_direction == "y":
                q_factor = cos(self.angle)
            elif self.q_direction == "element" or self.q_direction is None:
                q_factor = 1
            elif self.q_direction is not None:
                raise FEMException('Wrong parameters',
                                   "q-loads direction is not set property. Please choose 'x', 'y', or 'element'")
            q = self.q_load * q_factor

        return q + self.dead_load * cos(self.angle)

    @property
    def node_1(self):
        return self.node_map[self.node_id1]

    @property
    def node_2(self):
        return self.node_map[self.node_id2]

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)
        return self.element_force_vector

    def compile_stiffness_matrix(self):
        self.stiffness_matrix = stiffness_matrix(self.constitutive_matrix, self.kinematic_matrix)

    def compile_kinematic_matrix(self, a1, a2, l):
        self.kinematic_matrix = kinematic_matrix(a1, a2, l)

    def compile_constitutive_matrix(self, EA, EI, l):
        self.constitutive_matrix = constitutive_matrix(EA, EI, l, self.springs)

    def update_stiffness(self, factor, node):
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
        self.stiffness_matrix += geometric_stiffness_matrix(self.l, self.N_1, self.a1, self.a2)

    def reset(self):
        self.element_displacement_vector = np.zeros(6)
        self.element_primary_force_vector = np.zeros(6)

    def __add__(self, other):
        if self.id != other.id:
            raise FEMException('Wrong element:', 'only elements with the same id can be added.')
        el = copy.deepcopy(self)
        for unit in ['bending_moment', 'shear_force', 'deflection', 'extension', 'N_1', 'N_2']:
            if getattr(el, unit) is None:
                setattr(el, unit, getattr(other, unit))
            else:
                setattr(el, unit, getattr(el, unit) + getattr(other, unit))
        el.max_deflection = other.max_deflection if el.max_deflection is None else \
            max(el.max_deflection, other.max_deflection)

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
def kinematic_matrix(a1, a2, l):
    """
    Kinematic matrix of an element dependent of the angle ai and the length of the element.

    :param a1: (float) angle with respect to the x axis.
    :param l: (float) Length
    """
    c1 = cos(a1)
    s1 = sin(a1)
    c2 = cos(a2)
    s2 = sin(a2)
    return np.array([[-c1, s1, 0, c2, -s2, 0],
                     [s1 / l, c1 / l, -1, -s2 / l, -c2 / l, 0],
                     [-s1 / l, -c1 / l, 0, s2 / l, c2 / l, 1]])


def constitutive_matrix(EA, EI, l, spring=None):
    """
    :param EA: (float) Young's modules * Area
    :param EI: (float) Young's modules * Moment of Inertia
    :param l: (float) Length
    :param spring: (int) 1 or 2. Apply a hinge on the first of the second node.
    :return: (array)
    """
    matrix = np.array([[EA / l, 0, 0],
                      [0, 4 * EI / l, -2 * EI / l],
                      [0, -2 * EI / l, 4 * EI / l]])

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
        if 1 in spring:
            if spring[1] == 0:  # hinge
                matrix[1][1] = matrix[1][2] = matrix[2][1] = 0
            else:
                matrix[1][1] = 1 / (1 / matrix[1][1] + 1 / spring[1])
                matrix[2][1] = 1 / (1 / matrix[2][1] + 1 / spring[1])
        if 2 in spring:
            if spring[2] == 0:  # hinge
                matrix[1][2] = matrix[2][1] = matrix[2][2] = 0
            else:
                matrix[2][1] = 1 / (1 / matrix[2][1] + 1 / spring[2])
                matrix[1][2] = 1 / (1 / matrix[1][2] + 1 / spring[2])
    return matrix


def stiffness_matrix(var_constitutive_matrix, var_kinematic_matrix):
    kinematic_transposed_times_constitutive = np.dot(var_kinematic_matrix.transpose(), var_constitutive_matrix)
    return np.dot(kinematic_transposed_times_constitutive, var_kinematic_matrix)


def geometric_stiffness_matrix(l, N, a1, a2):
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
    return N/l * np.array([[6/5*s1**2, -6/5*s1*c1, -l/10*s1, -6/5*s2**2, 6/5*s2*c2, -l/10*s2],
                    [-6/5*s1*c1, 6/5*c1**2, l/10*c1, 6/5*s2*c2, -6/5*c2**2, l/10*c2],
                    [-l/10*s1, l/10*c1, 2*l**2/15, l/10*s2, -l/10*c2, -l**2/30],
                    [-6/5*s1**2, 6/5*s1*c1, l/10*s1, 6/5*s2**2, -6/5*s1*c2, l/10*s2],
                    [6/5*s1*c1, -6/5*c1**2, -l/10*c1, -6/5*s2*c2, 6/5*c2**2, -l/10*c2],
                    [-l/10*s1, l/10*c1, -l**2/30, l/10*s2, -l/10*c2, 2*l**2/15]]) \
                * np.array([1, -1, 1, 1, -1, 1])  # conversion from coordinate system


@lru_cache(CACHE_BOUND)
def det_axial(EA, L, q, x):
    """
    See notebook in: anastruct/fem/background/distributed_ax_force.ipynb

    :param q: (flt)
    :param x: (flt) Location of the axial force
    :param EA: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EA * (L * q / (2 * EA) - q * x / EA)

