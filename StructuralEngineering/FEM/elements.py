import math
import numpy as np
from StructuralEngineering.FEM.node import Node

"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""


class Element:
    def __init__(self, id, EA, EI, l, ai, point_1, point_2, hinge=None):
        """
        :param id: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param ai: angle between element and x-axis
        :param point_1: point object
        :param point_2: point object
        :param hinge: (integer) 1 or 2. Adds an hinge ad the first or second node.
        """
        self.id = id
        self.type = None
        self.EA = EA
        self.EI = EI
        self.l = l
        self.point_1 = point_1  # location
        self.point_2 = point_2  # location
        self.alpha = ai
        self.kinematic_matrix = kinematic_matrix(ai, l)
        self.constitutive_matrix = constitutive_matrix(EA, EI, l, hinge)
        self.stiffness_matrix = None
        self.node_id1 = None  # int
        self.node_id2 = None  # int
        self.node_ids = []
        self.node_1 = None  # node object
        self.node_2 = None  # node object
        self.element_displacement_vector = np.empty(6)
        self.element_primary_force_vector = np.zeros(6)  # acting external forces
        self.element_force_vector = None
        self.q_load = None
        self.N = None
        self.bending_moment = None
        self.shear_force = None
        self.deflection = None
        self.extension = None
        self.max_deflection = None
        self.compile_stifness_matrix()
        self.nodes_plastic = [False, False]

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)
        return self.element_force_vector

    def compile_stifness_matrix(self):
        self.stiffness_matrix = stiffness_matrix(self.constitutive_matrix, self.kinematic_matrix)

    def compile_kinematic_matrix(self, ai, l):
        self.kinematic_matrix = kinematic_matrix(ai, l)

    def update_stiffness(self, factor, node):
        if node == 1:
            self.constitutive_matrix[1][1] *= factor
            self.constitutive_matrix[1][2] *= factor
            self.constitutive_matrix[2][1] *= factor
        elif node == 2:
            self.constitutive_matrix[1][2] *= factor
            self.constitutive_matrix[2][1] *= factor
            self.constitutive_matrix[2][2] *= factor
        self.compile_stifness_matrix()


def kinematic_matrix(ai, l):
    """
    Kinematic matrix of an element dependent of the angle ai and the length of the element.

    :param ai: (float) angle with respect to the x axis.
    :param l: (float) Length
    """
    return np.array([[-math.cos(ai), math.sin(ai), 0, math.cos(ai), -math.sin(ai), 0],
                     [math.sin(ai) / l, math.cos(ai) / l, -1, -math.sin(ai) / l, -math.cos(ai) / l, 0],
                     [-math.sin(ai) / l, -math.cos(ai) / l, 0, math.sin(ai) / l, math.cos(ai) / l, 1]])


def constitutive_matrix(EA, EI, l, hinge=None):
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
    if hinge is None:
        return matrix
    elif hinge == 2:
        matrix[1][2] = matrix[2][1] = matrix[2][2] = 1e-14
    elif hinge == 1:
        matrix[1][1] = matrix[1][2] = matrix[2][1] = 1e-14
    return matrix


def stiffness_matrix(var_constitutive_matrix, var_kinematic_matrix):

    kinematic_transposed_times_constitutive = np.dot(var_kinematic_matrix.transpose(), var_constitutive_matrix)
    return np.dot(kinematic_transposed_times_constitutive, var_kinematic_matrix)