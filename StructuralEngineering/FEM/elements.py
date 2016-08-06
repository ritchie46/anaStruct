import math
import numpy as np
from StructuralEngineering.FEM.node import Node

"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""


class Element:
    def __init__(self, ID, EA, EI, l, ai, aj, point_1, point_2):
        """
        :param ID: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param ai: angle between element and x-axis
        :param aj: = ai
        :param point_1: point object
        :param point_2: point object
        """
        self.ID = ID
        self.EA = EA
        self.EI = EI
        self.l = l
        self.point_1 = point_1
        self.point_2 = point_2
        self.alpha = ai
        self.kinematic_matrix = kinematic_matrix(ai, aj, l)
        self.constitutive_matrix = constitutive_matrix(EA, EI, l)
        self.stiffness_matrix = stiffness_matrix(self.constitutive_matrix, self.kinematic_matrix)
        self.nodeID1 = None
        self.nodeID2 = None
        self.node_ids = []
        self.node_1 = None
        self.node_2 = None
        self.element_displacement_vector = np.empty(6)
        self.element_primary_force_vector = np.zeros(6)
        self.element_force_vector = None
        self.q_load = None
        self.N = None
        self.bending_moment = None
        self.shear_force = None

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)
        return self.element_force_vector


def kinematic_matrix(ai, aj, l):

    return np.array([[-math.cos(ai), math.sin(ai), 0, math.cos(aj), -math.sin(aj), 0],
                     [math.sin(ai) / l, math.cos(ai) / l, -1, -math.sin(aj) / l, -math.cos(aj) / l, 0],
                     [-math.sin(ai) / l, -math.cos(ai) / l, 0, math.sin(aj) / l, math.cos(aj) / l, 1]])


def constitutive_matrix(EA, EI, l):

    return np.array([[EA / l, 0, 0],
                     [0, 4 * EI / l, -2 * EI / l],
                     [0, -2 * EI / l, 4 * EI / l]])


def stiffness_matrix(var_constitutive_matrix, var_kinematic_matrix):

    kinematic_transposed_times_constitutive = np.dot(var_kinematic_matrix.transpose(), var_constitutive_matrix)
    return np.dot(kinematic_transposed_times_constitutive, var_kinematic_matrix)