import matplotlib.pyplot as plt
import numpy as np

from StructuralEngineering.intersection import do_intersect
from StructuralEngineering.trigonometry import Point


class Epsilon:
    def __init__(self):
        self.eps_array = None

    def create_eps_array(self, top_eps=0, bottom_eps=0, nValues=1000 ):
        self.eps_array = np.linspace(top_eps, bottom_eps, nValues)
        return self.eps_array


class CrossSection:
    def __init__(self, coordinate_list):
        self.nValues = 1000
        self.width_array = None
        self.height_array = None
        self.height = None
        self.center_of_gravity = None
        self.moment_of_inertia = None
        self.coordinate_list = None
        self.xValuesPerSection = np.zeros([self.nValues, 2])
        self.section_height = None

        self.section_coordinates(coordinate_list)
        self.det_center_of_gravity()
        self.det_moment_of_inertia()

    def create_height_array(self):
        self.height_array = np.linspace(self.height, 0, self.nValues)

    def section_coordinates(self, coordinate_list):
        """
        :param coordinate_list: [[x, z], [x,z]]
        :return: create an array that represents the width over the cross section.
        """

        # Closes the cross section if necessary
        if coordinate_list[0] != coordinate_list[-1]:
            coordinate_list.append(coordinate_list[0])

        self.coordinate_list = []
        for i in range(len(coordinate_list)):
            self.coordinate_list.append(Point(coordinate_list[i][0], coordinate_list[i][1]))

        # Check the input
        self.valid_cross_section()

        # The empty array that will be filled with width values
        self.width_array = np.empty(self.nValues)
        count = 0

        # determine the max z-value in the polygon and set this to the height
        self.height = 0
        for i in coordinate_list:
            if i[1] > self.height:
                self.height = i[1]

        self.create_height_array()

        # interpolate between the given points.
        for section_height in self.height_array:
            for val in range(len(coordinate_list)):
                if coordinate_list[val][1] >= section_height:
                    left_higher_coordinate = coordinate_list[val]
                    left_lower_coordinate = coordinate_list[val - 1]
                    break

            for val in range(1, len(coordinate_list)):
                if coordinate_list[-val][1] >= section_height:
                    right_higher_coordinate = coordinate_list[-val]
                    right_lower_coordinate = coordinate_list[-val + 1]
                    break

            diff_height_left = left_higher_coordinate[1] - left_lower_coordinate[1]
            diff_width_left = left_higher_coordinate[0] - left_lower_coordinate[0]
            diff_height_right = right_higher_coordinate[1] - right_lower_coordinate[1]
            diff_width_right = right_higher_coordinate[0] - right_lower_coordinate[0]

            if diff_height_left and diff_height_right > 1e-6:

                fraction = (section_height - left_lower_coordinate[1]) / diff_height_left
                x_left = diff_width_left * fraction + left_lower_coordinate[0]

                fraction = (section_height - right_lower_coordinate[1]) / diff_height_right
                x_right = diff_width_right * fraction + right_lower_coordinate[0]
                self.width_array[count] = (x_left + x_right)
                self.xValuesPerSection[count] = np.array([x_left, x_right])
            count += 1

        # set the height of one section
        self.section_height = self.height / self.nValues

    def det_center_of_gravity(self):
        sigA = 0
        sig_A_times_z = 0
        for i in range(self.nValues):
            b = self.width_array[i]
            if i < self.nValues - 1:
                h = self.height_array[i] - self.height_array[i + 1]
            else:
                h = self.height_array[-1] - self.height_array[-2]

            A = b * h
            sigA += A
            z = self.height_array[i] - 0.5 * h
            sig_A_times_z += A * z
        self.center_of_gravity = sig_A_times_z / sigA

        return self.center_of_gravity

    def det_moment_of_inertia(self):
        if not self.center_of_gravity:
            self.det_center_of_gravity()

        moment_of_inertia = 0
        for i in range(self.nValues):
            b = self.width_array[i]
            if i < self.nValues - 1:
                h = self.height_array[i] - self.height_array[i + 1]
            else:
                h = self.height_array[-1] - self.height_array[-2]

            z_section = self.height_array[i] - 0.5 * h
            inertia = 1 / 12 * b * h**3
            steiner = b * h * (self.center_of_gravity - z_section) ** 2

            moment_of_inertia += inertia + steiner
        self.moment_of_inertia = moment_of_inertia
        return moment_of_inertia

    def print_in_lines(self):
        minVal = 1e6
        maxVal = -1e6
        for i in range(self.nValues):

            # print 1 in the 100 lines
            if i % int(self.nValues / 100) == 0:
                # set minVal and maxVal in order to realize some whitespace in the plotting figure
                if self.xValuesPerSection[i][0] < minVal:
                    minVal = self.xValuesPerSection[i][0]
                if self.xValuesPerSection[i][1] > maxVal:
                    maxVal = self.xValuesPerSection[i][1]

                x = [self.xValuesPerSection[i][0], self.xValuesPerSection[i][1]]
                z = [self.height_array[i], self.height_array[i]]
                plt.plot(x, z, color='b')
        plt.axis([minVal - 100, maxVal + 100, -100, self.height_array[0] + 100])
        plt.show()

    def valid_cross_section(self):
        """
        check if the cross-section is valid. Lines may not intersect
        :return: boolean
        """

        # Make from every two unique points a line
        lines = []
        for i in range(1, len(self.coordinate_list)):
            point_1 = self.coordinate_list[i - 1]
            point_2 = self.coordinate_list[i]

            # add every 2 points, after the first 2 points as aline
            lines.append([point_1, point_2])

        doesIntersect = False
        # Last line connects the polygon, thus will result to an intersection True
        lines.__delitem__(-1)
        for i in range(len(lines)):

            # Check for every line if it intersects with other lines, except with the two connected lines
            for val in range(len(lines)):
                if val != i and val != i - 1 and val != i + 1:
                    if do_intersect(lines[val], lines[i]):
                        doesIntersect = True
        assert (not doesIntersect), "The lines of the cross section have one or multiple intersections"
        return not doesIntersect









