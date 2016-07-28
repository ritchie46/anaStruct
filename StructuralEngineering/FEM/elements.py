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

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)

        return self.element_force_vector

    def determine_node_results(self):
        self.node_1 = Node(
            ID=self.node_ids[0],
            Fx=self.element_force_vector[0] + self.element_primary_force_vector[0],
            Fz=self.element_force_vector[1] + self.element_primary_force_vector[1],
            Ty=self.element_force_vector[2] + self.element_primary_force_vector[2],
            ux=self.element_displacement_vector[0],
            uz=self.element_displacement_vector[1],
            phiy=self.element_displacement_vector[2],
        )

        self.node_2 = Node(
            ID=self.node_ids[1],
            Fx=self.element_force_vector[3] + self.element_primary_force_vector[3],
            Fz=self.element_force_vector[4] + self.element_primary_force_vector[4],
            Ty=self.element_force_vector[5] + self.element_primary_force_vector[5],
            ux=self.element_displacement_vector[3],
            uz=self.element_displacement_vector[4],
            phiy=self.element_displacement_vector[5]
        )
        self.determine_normal_force()

    def determine_normal_force(self):
        dx = self.point_1.x - self.point_2.x

        if math.isclose(dx, 0):  # element is vertical
            if self.point_1.z < self.point_2.z:  # point 1 is bottom
                self.N = -self.node_1.Fz  # compression and tension in opposite direction
            else:
                self.N = self.node_1.Fz  # compression and tension in opposite direction

        elif math.isclose(self.alpha, 0):  # element is horizontal
            if self.point_1.x < self.point_2.x:  # point 1 is left
                self.N = -self.node_1.Fx  # compression and tension in good direction
            else:
                self.N = self.node_1.Fx  # compression and tension in opposite direction

        else:
            if math.cos(self.alpha) > 0:
                if self.point_1.x < self.point_2.x:  # point 1 is left
                    if self.node_1.Fx > 0:  # compression in element
                        self.N = -math.sqrt(self.node_1.Fx**2 + self.node_1.Fz**2)
                    else:
                        self.N = math.sqrt(self.node_1.Fx ** 2 + self.node_1.Fz ** 2)
                else:  # point 1 is right
                    if self.node_1.Fx < 0:  # compression in element
                        self.N = -math.sqrt(self.node_1.Fx ** 2 + self.node_1.Fz ** 2)
                    else:
                        self.N = math.sqrt(self.node_1.Fx ** 2 + self.node_1.Fz ** 2)

    def plot_values_normal_force(self, factor):
        x1 = self.point_1.x
        y1 = -self.point_1.z
        x2 = self.point_2.x
        y2 = -self.point_2.z

        x_1 = x1 + self.N * math.cos(0.5 * math.pi + self.alpha) * factor
        y_1 = y1 + self.N * math.sin(0.5 * math.pi + self.alpha) * factor
        x_2 = x2 + self.N * math.cos(0.5 * math.pi + self.alpha) * factor
        y_2 = y2 + self.N * math.sin(0.5 * math.pi + self.alpha) * factor

        x_val = [x1, x_1, x_2, x2]
        y_val = [y1, y_1, y_2, y2]
        return x_val, y_val

    def plot_values_element(self):
        x_val = [self.point_1.x, self.point_2.x]
        y_val = [-self.point_1.z, -self.point_2.z]
        return x_val, y_val, self.ID

    def plot_values_bending_moment(self, factor, con):
        """
        :param factor: scaling the plot
        :param con: amount of x-values
        :return:
        """
        # plot the bending moment
        x1 = self.point_1.x
        y1 = -self.point_1.z
        x2 = self.point_2.x
        y2 = -self.point_2.z
        dx = x2 - x1
        dy = y2 - y1

        if math.isclose(self.alpha, 0.5 * math.pi, rel_tol=1e-4) or\
                math.isclose(self.alpha, 1.5 * math.pi, rel_tol=1e-4):
            # point 1 is bottom
            x_1 = x1 + self.node_1.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
            y_1 = y1 + self.node_1.Ty * math.sin(0.5 * math.pi + self.alpha) * factor
            x_2 = x2 - self.node_2.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
            y_2 = y2 - self.node_2.Ty * math.sin(0.5 * math.pi + self.alpha) * factor

        else:
            if dx > 0:  # point 1 = left and point 2 is right
                # With a negative moment (at support), left is positive and right is negative
                # node 1 is left, node 2 is right
                x_1 = x1 + self.node_1.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
                y_1 = y1 + self.node_1.Ty * math.sin(0.5 * math.pi + self.alpha) * factor
                x_2 = x2 - self.node_2.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
                y_2 = y2 - self.node_2.Ty * math.sin(0.5 * math.pi + self.alpha) * factor
            else:
                # node 1 is right, node 2 is left
                x_1 = x1 - self.node_1.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
                y_1 = y1 - self.node_1.Ty * math.sin(0.5 * math.pi + self.alpha) * factor
                x_2 = x2 + self.node_2.Ty * math.cos(0.5 * math.pi + self.alpha) * factor
                y_2 = y2 + self.node_2.Ty * math.sin(0.5 * math.pi + self.alpha) * factor

        x_val = np.linspace(0, 1, con)
        y_val = np.empty(con)
        dx = x_2 - x_1
        dy = y_2 - y_1
        count = 0

        for i in x_val:
            x_val[count] = x_1 + i * dx
            y_val[count] = y_1 + i * dy

            if self.q_load:
                x = i * self.l
                q_part = (-0.5 * -self.q_load * x**2 + 0.5 * -self.q_load * self.l * x)

                y_val[count] += math.cos(self.alpha) * q_part * factor
            count += 1

        x_val = np.append(x_val, x2)
        y_val = np.append(y_val, y2)
        x_val = np.insert(x_val, 0, x1)
        y_val = np.insert(y_val, 0, y1)

        if self.q_load:
            sagging_moment = (-self.node_1.Ty + self.node_2.Ty) * 0.5 + 0.125 * self.q_load * self.l**2
            return x_val, y_val, sagging_moment
        else:
            return x_val, y_val

    def plot_values_shear_force(self, factor=1):
        dx = self.point_1.x - self.point_2.x
        x1 = self.point_1.x
        y1 = -self.point_1.z
        x2 = self.point_2.x
        y2 = -self.point_2.z

        if math.isclose(dx, 0):  # element is vertical
            shear_1 = -(math.sin(self.alpha) * self.node_1.Fx + math.cos(self.alpha) * self.node_1.Fz)
            shear_2 = (math.sin(self.alpha) * self.node_2.Fx + math.cos(self.alpha) * self.node_2.Fz)

            if self.point_1.z > self.point_2.z:  # point 1 is bottom
                x_1 = x1 + shear_1 * math.cos(1.5 * math.pi + self.alpha) * factor
                y_1 = y1 + shear_1 * math.sin(1.5 * math.pi + self.alpha) * factor
                x_2 = x2 + shear_2 * math.cos(1.5 * math.pi + self.alpha) * factor
                y_2 = y2 + shear_2 * math.sin(1.5 * math.pi + self.alpha) * factor
            else:
                x_1 = x1 + shear_1 * math.cos(0.5 * math.pi + self.alpha) * factor
                y_1 = y1 + shear_1 * math.sin(0.5 * math.pi + self.alpha) * factor
                x_2 = x2 + shear_2 * math.cos(0.5 * math.pi + self.alpha) * factor
                y_2 = y2 + shear_2 * math.sin(0.5 * math.pi + self.alpha) * factor

        elif math.isclose(self.alpha, 0):  # element is horizontal
            if self.point_1.x < self.point_2.x:  # point 1 is left
                shear_1 = -(math.sin(self.alpha) * self.node_1.Fx + math.cos(self.alpha) * self.node_1.Fz)
                shear_2 = math.sin(self.alpha) * self.node_2.Fx + math.cos(self.alpha) * self.node_2.Fz
                x_1 = x1 + shear_1 * math.cos(1.5 * math.pi + self.alpha) * factor
                y_1 = y1 + shear_1 * math.sin(1.5 * math.pi + self.alpha) * factor
                x_2 = x2 + shear_2 * math.cos(1.5 * math.pi + self.alpha) * factor
                y_2 = y2 + shear_2 * math.sin(1.5 * math.pi + self.alpha) * factor
            else:
                shear_1 = math.sin(self.alpha) * self.node_1.Fx + math.cos(self.alpha) * self.node_1.Fz
                shear_2 = -(math.sin(self.alpha) * self.node_2.Fx + math.cos(self.alpha) * self.node_2.Fz)
                x_1 = x1 + shear_1 * math.cos(0.5 * math.pi + self.alpha) * factor
                y_1 = y1 + shear_1 * math.sin(0.5 * math.pi + self.alpha) * factor
                x_2 = x2 + shear_2 * math.cos(0.5 * math.pi + self.alpha) * factor
                y_2 = y2 + shear_2 * math.sin(0.5 * math.pi + self.alpha) * factor

        else:
            if math.cos(self.alpha) > 0:
                if self.point_1.x < self.point_2.x:  # point 1 is left
                    shear_1 = -(math.sin(self.alpha) * self.node_1.Fx + math.cos(self.alpha) * self.node_1.Fz)
                    shear_2 = math.sin(self.alpha) * self.node_2.Fx + math.cos(self.alpha) * self.node_2.Fz
                    x_1 = x1 + shear_1 * math.cos(1.5 * math.pi + self.alpha) * factor
                    y_1 = y1 + shear_1 * math.sin(1.5 * math.pi + self.alpha) * factor
                    x_2 = x2 + shear_2 * math.cos(1.5 * math.pi + self.alpha) * factor
                    y_2 = y2 + shear_2 * math.sin(1.5 * math.pi + self.alpha) * factor
                else:  # point 1 is right
                    shear_1 = math.sin(self.alpha) * self.node_1.Fx + math.cos(self.alpha) * self.node_1.Fz
                    shear_2 = -(math.sin(self.alpha) * self.node_2.Fx + math.cos(self.alpha) * self.node_2.Fz)
                    x_1 = x1 + shear_1 * math.cos(0.5 * math.pi + self.alpha) * factor
                    y_1 = y1 + shear_1 * math.sin(0.5 * math.pi + self.alpha) * factor
                    x_2 = x2 + shear_2 * math.cos(0.5 * math.pi + self.alpha) * factor
                    y_2 = y2 + shear_2 * math.sin(0.5 * math.pi + self.alpha) * factor

        x_val = [x1, x_1, x_2, x2]
        y_val = [y1, y_1, y_2, y2]
        return x_val, y_val, shear_1, shear_2


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