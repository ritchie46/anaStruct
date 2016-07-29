import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Plotter:
    def __init__(self, system):
        self.system = system
        fig = plt.figure()
        self.one_fig = fig.add_subplot(111)
        plt.tight_layout()
        plt.style.use('ggplot')
        self.max_val = None
        self.plot_structure()
        self.max_force = 0

    def __set_factor(self, value_1, value_2):
        """
        :param value_1: value of the force/ moment at point 1
        :param value_2: value of the force/ moment at point 2
        :return: factor for scaling the force/moment in the plot
        """

        if abs(value_1) > self.max_force:
            self.max_force = abs(value_1)
        if abs(value_2) > self.max_force:
            self.max_force = abs(value_2)

        if math.isclose(self.max_force, 0):
            factor = 0.1
        else:
            factor = 1 / self.max_force
        return factor

    def __fixed_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """

        width = 4 / max_val
        height = 4 / max_val
        for node in self.system.supports_fixed:

            support_patch = mpatches.Rectangle((node.point.x - width * 0.25, -node.point.z - width * 0.25),
                                               width, height, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __hinged_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 3 / max_val
        for node in self.system.supports_hinged:
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                    numVertices=3, radius=radius, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __roll_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 3 / max_val
        for node in self.system.supports_roll:
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                    numVertices=3, radius=radius, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)
            y = node.point.z - 2 * radius
            self.one_fig.plot([node.point.x - radius, node.point.x + radius], [y, y], color='r')

    def plot_structure(self, plot_now=False):
        """
        :param plot_now: boolean if True, plt.figure will plot.
        :return: -
        """
        max_val = 0
        for el in self.system.elements:
            # plot structure
            axis_values = el.plot_values_element()
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.one_fig.plot(x_val, y_val, color='black', marker='s')

            if max(max(x_val), max(y_val)) > max_val:
                max_val = max(max(x_val), max(y_val))

            # add node ID to plot
            self.one_fig.text(x_val[0] + 0.1, y_val[0] + 0.1, '%d' % el.nodeID1, color='g', fontsize=9, zorder=10)
            self.one_fig.text(x_val[-1] + 0.1, y_val[-1] + 0.1, '%d' % el.nodeID2, color='g', fontsize=9, zorder=10)

            # add element ID to plot
            if 0 < el.alpha <= math.pi:
                x_val = (x_val[0] + x_val[-1]) / 2 + math.sin(el.alpha) * 0.1
            else:
                x_val = (x_val[0] + x_val[-1]) / 2 - math.sin(el.alpha) * 0.1
            y_val = (y_val[0] + y_val[-1]) / 2 + math.cos(el.alpha) * 0.1

            self.one_fig.text(x_val, y_val, "%d" % el.ID, color='r', fontsize=9, zorder=10)

        self.max_val = max_val
        max_val += 2
        self.one_fig.axis([-2, max_val, -2, max_val])

        # add supports
        self.__fixed_support_patch(max_val)
        self.__hinged_support_patch(max_val)
        self.__roll_support_patch(max_val)

        if plot_now:
            plt.show()

    def plot_force(self, axis_values, force_1, force_2):

            # plot force
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.one_fig.plot(x_val, y_val, color='b')

            # add value to plot
            self.one_fig.text(x_val[1] + 2 / self.max_val, y_val[1] + 2 / self.max_val, "%s" % abs(round(force_1, 1)),
                              fontsize=9)
            self.one_fig.text(x_val[-2] - 2 / self.max_val, y_val[-2] + 2 / self.max_val, "%s" % abs(round(force_2, 1)),
                              fontsize=9)

    def normal_force(self):
        # determine max factor for scaling
        for el in self.system.elements:
            factor = self.__set_factor(el.N, el.N)

        for el in self.system.elements:
            axis_values = el.plot_values_normal_force(factor)
            self.plot_force(axis_values, el.N, el.N)

        plt.show()

    def bending_moment(self):
        con = 20
        # determine max factor for scaling
        for el in self.system.elements:
            factor = self.__set_factor(el.node_1.Ty, el.node_2.Ty)

        for el in self.system.elements:
            axis_values = el.plot_values_bending_moment(factor, con)
            self.plot_force(axis_values, el.node_1.Ty, el.node_2.Ty)

            if el.q_load:
                index = con // 2
                self.one_fig.text(axis_values[0][index], axis_values[1][index], "%s" % round(abs(axis_values[2]), 1))
        plt.show()

    def shear_force(self):

        # determine max factor for scaling
        for el in self.system.elements:
            sol = el.plot_values_shear_force(factor=1)
            shear_1 = sol[-2]
            shear_2 = sol[-1]
            factor = self.__set_factor(shear_1, shear_2)

        for el in self.system.elements:
            axis_values = el.plot_values_shear_force(factor)
            shear_1 = axis_values[-2]
            shear_2 = axis_values[-1]
            self.plot_force(axis_values, shear_1, shear_2)
        plt.show()
