import math
import numpy as np
from StructuralEngineering.FEM.node import Node

"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""


class Element:
    def __init__(self, id, EA, EI, l, ai, vertex_1, vertex_2, spring=None):
        """
        :param id: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param ai: angle between element and x-axis
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
        self.id = id
        self.type = None
        self.EA = EA
        self.EI = EI
        self.l = l
        self.springs = spring
        self.vertex_1 = vertex_1  # location
        self.vertex_2 = vertex_2  # location
        self.ai = ai
        self.kinematic_matrix = kinematic_matrix(ai, ai, l)
        self.constitutive_matrix = None
        self.stiffness_matrix = None
        self.node_id1 = None  # int
        self.node_id2 = None  # int
        self.node_ids = []
        self.node_1 = None  # node object
        self.node_2 = None  # node object
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
                q_factor = math.sin(self.ai)
            elif self.q_direction == "y":
                q_factor = math.cos(self.ai)
            else:
                q_factor = 1
            q = self.q_load * q_factor

        return q + self.dead_load * math.cos(self.ai)

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)
        return self.element_force_vector

    def compile_stiffness_matrix(self):
        self.stiffness_matrix = stiffness_matrix(self.constitutive_matrix, self.kinematic_matrix)

    def compile_kinematic_matrix(self, ai, aj, l):
        self.kinematic_matrix = kinematic_matrix(ai, aj, l)

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

    def reset(self):
        self.element_displacement_vector = np.zeros(6)
        self.element_primary_force_vector = np.zeros(6)


def kinematic_matrix(ai, aj, l):
    """
    Kinematic matrix of an element dependent of the angle ai and the length of the element.

    :param ai: (float) angle with respect to the x axis.
    :param l: (float) Length
    """
    return np.array([[-math.cos(ai), math.sin(ai), 0, math.cos(aj), -math.sin(aj), 0],
                     [math.sin(ai) / l, math.cos(ai) / l, -1, -math.sin(aj) / l, -math.cos(aj) / l, 0],
                     [-math.sin(ai) / l, -math.cos(ai) / l, 0, math.sin(aj) / l, math.cos(aj) / l, 1]])


def constitutive_matrix(EA, EI, l, spring=None):
    """
    :param EA: (float) Young's modules * Area
    :param EI: (float) Young's modules * Moment of Inertia
    :param l: (float) Length
    :param hinge: (int) 1 or 2. Apply a hinge on the first of the second node.
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


def det_moment(kl, kr, q, x, EI, L):
    """
    See notebook in: StructuralEngineering/FEM/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI*(-L**3*kl*q*(6*EI + L*kr)/(12*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) +
               L*q*x*(12*EI**2 + 5*EI*L*kl + 3*EI*L*kr +
                      L**2*kl*kr)/(2*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) - q*x**2/(2*EI))


def det_shear(kl, kr, q, x, EI, L):
    """
    See notebook in: StructuralEngineering/FEM/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI*(L*q*(12*EI**2 + 5*EI*L*kl + 3*EI*L*kr + L**2*kl*kr) /
               (2*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) - q*x/EI)


def det_axial(EA, L, q, x):
    """
    See notebook in: StructuralEngineering/FEM/background/distributed_ax_force.ipynb

    :param q: (flt)
    :param x: (flt) Location of the axial force
    :param EA: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    EA * (L * q / (2 * EA) - q * x / EA)