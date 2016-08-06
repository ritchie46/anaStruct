import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Plotter:
    def __init__(self, system):
        self.system = system
        self.max_val = None
        self.max_force = 0

    def __start_plot(self):
        fig = plt.figure()
        self.one_fig = fig.add_subplot(111)
        plt.tight_layout()

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

        width = 0.05 * max_val
        height = 0.05 * max_val
        for node in self.system.supports_fixed:

            support_patch = mpatches.Rectangle((node.point.x - width * 0.5, - node.point.z - width * 0.5),
                                               width, height, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __hinged_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.04 * max_val
        for node in self.system.supports_hinged:
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                    numVertices=3, radius=radius, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __roll_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.03 * max_val
        for node in self.system.supports_roll:
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                    numVertices=3, radius=radius, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)
            y = node.point.z - 2 * radius
            self.one_fig.plot([node.point.x - radius, node.point.x + radius], [y, y], color='r')

    def __rotating_spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.04 * max_val

        for node in self.system.supports_spring_y:
            r = np.arange(0, radius, 0.001)
            theta = 25 * math.pi * r / (0.2 * max_val)
            x_val = []
            y_val = []

            count = 0
            for angle in theta:
                x = math.cos(angle) * r[count] + node.point.x
                y = math.sin(angle) * r[count] - radius - node.point.z
                x_val.append(x)
                y_val.append(y)
                count += 1

            self.one_fig.plot(x_val, y_val, color='r', zorder=9)

            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius * 3),
                                                    numVertices=3, radius=radius * 0.9, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        h = 0.04 * max_val
        left = -0.5 * h
        right = 0.5 * h
        dh = 0.2 * h

        for node in self.system.supports_spring_z:
            yval = np.arange(0, -9, -1)
            yval = yval * dh
            xval = np.array([0, 0, left, right, left, right, left, 0, 0])

            yval = yval - node.point.z
            xval = xval + node.point.x
            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - h * 2.6),
                                                    numVertices=3, radius=h * 0.9, color='r', zorder=10)

            self.one_fig.plot(xval, yval, color='r', zorder=10)
            self.one_fig.add_patch(support_patch)

        for node in self.system.supports_spring_x:
            xval = np.arange(0, 9, 1)
            xval = xval * dh
            yval = np.array([0, 0, left, right, left, right, left, 0, 0])

            xval = xval + node.point.x
            yval = yval - node.point.z
            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x + h * 1.7, -node.point.z - h ),
                                                    numVertices=3, radius=h * 0.9, color='r', zorder=10)

            self.one_fig.plot(xval, yval, color='r', zorder=10)
            self.one_fig.add_patch(support_patch)

    def __q_load_patch(self, max_val):
        """
        :param max_val: max scale of the plot

        xn1;yn1  q-load   xn1;yn1
        -------------------
        |__________________|
        x1;y1  element    x2;y2
        """
        h = 0.05 * max_val

        for q_ID in self.system.loads_q:
            for el in self.system.elements:
                if el.ID == q_ID:
                    if el.q_load > 0:
                        direction = 1
                    else:
                        direction = -1

                    x1 = el.point_1.x
                    y1 = -el.point_1.z
                    x2 = el.point_2.x
                    y2 = -el.point_2.z
                    # - value, because the positive z of the system is opposite of positive y of the plotter
                    xn1 = x1 + math.sin(-el.alpha) * h * direction
                    yn1 = y1 + math.cos(-el.alpha) * h * direction
                    xn2 = x2 + math.sin(-el.alpha) * h * direction
                    yn2 = y2 + math.cos(-el.alpha) * h * direction
                    self.one_fig.plot([x1, xn1, xn2, x2], [y1, yn1, yn2, y2], color='g')

                    # arrow
                    xa_1 = (x2 - x1) * 0.2 + x1 + math.sin(-el.alpha) * 0.8 * h * direction
                    ya_1 = (y2 - y1) * 0.2 + y1 + math.cos(-el.alpha) * 0.8 * h * direction
                    len_x = math.sin(-el.alpha - math.pi) * 0.6 * h * direction
                    len_y = math.cos(-el.alpha - math.pi) * 0.6 * h * direction
                    xt = xa_1 + math.sin(-el.alpha) * 0.4 * h * direction
                    yt = ya_1 + math.cos(-el.alpha) * 0.4 * h * direction
                    # fc = face color, ec = edge color
                    self.one_fig.arrow(xa_1, ya_1, len_x, len_y, head_width=h*0.25, head_length=0.2*h, ec='g', fc='g')
                    self.one_fig.text(xt, yt, "q=%d" % el.q_load, color='k', fontsize=9, zorder=10)

    def __arrow_patch_values(self, Fx, Fz, node, h):
        """
        :param Fx: (float)
        :param Fz: (float)

        -- One of the above must be zero. The function created to find the non-zero F-direction.

        :param node: (Node object)
        :param h: (float) Is a scale variable
        :return: Variables for the matplotlib plotter
        """
        if Fx > 0:  # Fx is positive
            x = node.point.x - h
            y = -node.point.z
            len_x = 0.8 * h
            len_y = 0
            F = Fx
        elif Fx < 0:  # Fx is negative
            x = node.point.x + h
            y = -node.point.z
            len_x = -0.8 * h
            len_y = 0
            F = Fx
        elif Fz > 0:  # Fz is positive
            x = node.point.x
            y = -node.point.z + h
            len_x = 0
            len_y = -0.8 * h
            F = Fz
        else:  # Fz is negative
            x = node.point.x
            y = -node.point.z - h
            len_x = 0
            len_y = 0.8 * h
            F = Fz

        return x, y, len_x, len_y, F

    def __point_load_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        h = 0.1 * max_val

        for F_tuple in self.system.loads_point:
            for node in self.system.node_objects:
                if node.ID == F_tuple[0]:  # F_tuple[0] = ID
                    sol = self.__arrow_patch_values(F_tuple[1], F_tuple[2], node, h)
                    x = sol[0]
                    y = sol[1]
                    len_x = sol[2]
                    len_y = sol[3]
                    F = sol[4]

                    self.one_fig.arrow(x, y, len_x, len_y, head_width=h*0.15, head_length=0.2*h, ec='b', fc='orange',
                                       zorder=11)
                    self.one_fig.text(x, y, "F=%d" % F, color='k', fontsize=9, zorder=10)

    def plot_structure(self, plot_now=False, supports=True):
        """
        :param plot_now: (boolean) if True, plt.figure will plot.
        :param supports: (boolean) if True, supports are plotted.
        :return:
        """
        self.__start_plot()
        max_x = 0
        max_z = 0
        for el in self.system.elements:
            # plot structure
            axis_values = plot_values_element(el)
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.one_fig.plot(x_val, y_val, color='black', marker='s')

            # determine max values for scaling the plotter
            max_x = max([abs(x) for x in x_val]) if max([abs(x) for x in x_val]) > max_x else max_x
            max_z = max([abs(x) for x in y_val]) if max([abs(x) for x in y_val]) > max_z else max_z

            # add node ID to plot
            self.one_fig.text(x_val[0] + 0.1, y_val[0] + 0.1, '%d' % el.nodeID1, color='g', fontsize=9, zorder=10)
            self.one_fig.text(x_val[-1] + 0.1, y_val[-1] + 0.1, '%d' % el.nodeID2, color='g', fontsize=9, zorder=10)

        max_val = max(max_x, max_z)
        self.max_val = max_val
        offset = max_val // 2
        self.one_fig.axis([-offset, max_val + offset, -offset, max_val + offset])

        for el in self.system.elements:
            # add element ID to plot
            axis_values = plot_values_element(el)
            x_val = axis_values[0]
            y_val = axis_values[1]

            factor = 1.5 / self.max_val
            x_val = (x_val[0] + x_val[-1]) / 2 - math.sin(el.alpha) * factor
            y_val = (y_val[0] + y_val[-1]) / 2 + math.cos(el.alpha) * factor

            self.one_fig.text(x_val, y_val, "%d" % el.ID, color='r', fontsize=9, zorder=10)

        # add supports
        if supports:
            self.__fixed_support_patch(max_val)
            self.__hinged_support_patch(max_val)
            self.__roll_support_patch(max_val)
            self.__rotating_spring_support_patch(max_val)
            self.__spring_support_patch(max_val)

        if plot_now:
            # add_loads
            self.__q_load_patch(max_val)
            self.__point_load_patch(max_val)
            plt.show()

    def plot_force(self, axis_values, force_1, force_2):

            # plot force
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.one_fig.plot(x_val, y_val, color='b')

            # add value to plot
            self.one_fig.text(x_val[1] - 2 / self.max_val, y_val[1] + 2 / self.max_val, "%s" % round(force_1, 1),
                              fontsize=9, ha='center', va='center',)
            self.one_fig.text(x_val[-2] - 2 / self.max_val, y_val[-2] + 2 / self.max_val, "%s" % round(force_2, 1),
                              fontsize=9, ha='center', va='center',)

    def normal_force(self):
        self.plot_structure()

        # determine max factor for scaling
        for el in self.system.elements:
            factor = self.__set_factor(el.N, el.N)

        for el in self.system.elements:
            if math.isclose(el.N, 0, rel_tol=1e-5, abs_tol=1e-9):
                pass
            else:
                axis_values = plot_values_normal_force(el, factor)
                self.plot_force(axis_values, el.N, el.N)

                point = (el.point_2 - el.point_1) / 2 + el.point_1
                if el.N < 0:
                    point.displace_polar(alpha=el.alpha + 0.5 * math.pi, radius=0.5 * el.N * factor, inverse_z_axis=True)

                    self.one_fig.text(point.x, -point.z, "-", ha='center', va='center',
                                      fontsize=20, color='b')
                if el.N > 0:
                    point.displace_polar(alpha=el.alpha + 0.5 * math.pi, radius=0.5 * el.N * factor, inverse_z_axis=True)

                    self.one_fig.text(point.x, -point.z, "+", ha='center', va='center',
                                      fontsize=14, color='b')
        plt.show()

    def bending_moment(self):
        self.plot_structure()
        con = 20

        # determine max factor for scaling
        factor = 0
        for el in self.system.elements:
            if el.q_load:
                m_sag = (el.node_1.Ty - el.node_2.Ty) * 0.5 - 1 / 8 * el.q_load * el.l**2
                value_1 = max(abs(el.node_1.Ty), abs(m_sag))
                value_2 = max(value_1, abs(el.node_2.Ty))
                el_factor = self.__set_factor(value_1, value_2)
            else:
                el_factor = self.__set_factor(el.node_1.Ty, el.node_2.Ty)
            factor = el_factor if (el_factor > factor) else factor

        # determine the axis values
        for el in self.system.elements:
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and el.q_load is None:
                # If True there is no bending moment, so no need for plotting.
                pass

            else:
                axis_values = plot_values_bending_moment(el, factor, con)
                self.plot_force(axis_values, abs(el.node_1.Ty), abs(el.node_2.Ty))

                if el.q_load:
                    index = con // 2
                    offset = -self.max_val * 0.05
                    x = axis_values[0][index] + math.sin(-el.alpha) * offset
                    y = axis_values[1][index] + math.cos(-el.alpha) * offset
                    self.one_fig.text(x, y, "%s" % round(abs(axis_values[2]), 1),
                                      fontsize=9)
        plt.show()

    def shear_force(self):
        self.plot_structure()

        # determine max factor for scaling
        factor = 0
        for el in self.system.elements:
            shear_1 = max(el.shear_force)
            shear_2 = max(el.shear_force)
            el_factor = self.__set_factor(shear_1, shear_2)
            factor = el_factor if (el_factor > factor) else factor

        for el in self.system.elements:
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and el.q_load is None:
                # If True there is no bending moment, thus no shear force, so no need for plotting.
                pass
            else:
                axis_values = plot_values_shear_force(el, factor)
                shear_1 = axis_values[-2]
                shear_2 = axis_values[-1]
                self.plot_force(axis_values, shear_1, shear_2)
        plt.show()

    def reaction_force(self):
        self.plot_structure(supports=False)
        self.one_fig.text(self.max_val * 0.1 - 2, self.max_val * 0.9, "REACTION FORCE", fontsize=8)

        h = 0.2 * self.max_val
        max_force = 0

        for node in self.system.reaction_forces:
            max_force = abs(node.Fx) if abs(node.Fx) > max_force else max_force
            max_force = abs(node.Fz) if abs(node.Fz) > max_force else max_force

        for node in self.system.reaction_forces:
            if not math.isclose(node.Fx, 0, rel_tol=1e-5, abs_tol=1e-9):
                # x direction
                scale = abs(node.Fx) / max_force * h
                sol = self.__arrow_patch_values(node.Fx, 0, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.one_fig.arrow(x, y, len_x, len_y, head_width=h * 0.15, head_length=0.2 * scale, ec='b', fc='orange',
                                   zorder=11)
                self.one_fig.text(x, y, "R=%d" % node.Fx, color='k', fontsize=9, zorder=10)

            if not math.isclose(node.Fz, 0, rel_tol=1e-5, abs_tol=1e-9):
                # z direction
                scale = abs(node.Fz) / max_force * h
                sol = self.__arrow_patch_values(0, node.Fz, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.one_fig.arrow(x, y, len_x, len_y, head_width=h * 0.15, head_length=0.2 * scale, ec='b', fc='orange',
                                   zorder=11)
                self.one_fig.text(x, y, "R=%d" % node.Fz, color='k', fontsize=9, zorder=10)

            if not math.isclose(node.Ty, 0, rel_tol=1e-5, abs_tol=1e-9):
                """
                'r: regex
                '$...$': render the strings using mathtext
                """
                if node.Ty > 0:
                    self.one_fig.plot(node.point.x, -node.point.z, marker=r'$\circlearrowleft$', ms=25,
                                      color='orange')
                if node.Ty < 0:
                    self.one_fig.plot(node.point.x, -node.point.z, marker=r'$\circlearrowright$', ms=25,
                                      color='orange')

                self.one_fig.text(node.point.x + h * 0.2, -node.point.z + h * 0.2, "T=%d" % node.Ty, color='k',
                                  fontsize=9, zorder=10)
        plt.show()

"""
### element functions ###
"""


def plot_values_element(element):
    x_val = [element.point_1.x, element.point_2.x]
    y_val = [-element.point_1.z, -element.point_2.z]
    return x_val, y_val


def plot_values_shear_force(element, factor=1):
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    shear_1 = element.shear_force[0]
    shear_2 = element.shear_force[-1]

    # apply angle ai
    x_1 = x1 + shear_1 * math.sin(-element.alpha) * factor
    y_1 = y1 + shear_1 * math.cos(-element.alpha) * factor
    x_2 = x2 + shear_2 * math.sin(-element.alpha) * factor
    y_2 = y2 + shear_2 * math.cos(-element.alpha) * factor

    x_val = [x1, x_1, x_2, x2]
    y_val = [y1, y_1, y_2, y2]
    return x_val, y_val, shear_1, shear_2


def plot_values_normal_force(element, factor):
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    x_1 = x1 + element.N * math.cos(0.5 * math.pi + element.alpha) * factor
    y_1 = y1 + element.N * math.sin(0.5 * math.pi + element.alpha) * factor
    x_2 = x2 + element.N * math.cos(0.5 * math.pi + element.alpha) * factor
    y_2 = y2 + element.N * math.sin(0.5 * math.pi + element.alpha) * factor

    x_val = [x1, x_1, x_2, x2]
    y_val = [y1, y_1, y_2, y2]
    return x_val, y_val


def plot_values_bending_moment(element, factor, con):
    """
    :param element: (object) of the Element class
    :param factor: (float) scaling the plot
    :param con: (integer) amount of x-values
    :return:
    """
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    # Determine forces for horizontal element ai = 0
    T_left = element.node_1.Ty
    T_right = -element.node_2.Ty

    # apply angle ai
    x1 += T_left * math.sin(-element.alpha) * factor
    y1 += T_left * math.cos(-element.alpha) * factor
    x2 += T_right * math.sin(-element.alpha) * factor
    y2 += T_right * math.cos(-element.alpha) * factor

    x_val = np.linspace(0, 1, con)
    y_val = np.empty(con)
    dx = x2 - x1
    dy = y2 - y1

    # determine moment for 0 < x < length of the element
    count = 0
    for i in x_val:
        x_val[count] = x1 + i * dx
        y_val[count] = y1 + i * dy

        if element.q_load:
            x = i * element.l
            q_part = (-0.5 * -element.q_load * x**2 + 0.5 * -element.q_load * element.l * x)

            x_val[count] += math.sin(-element.alpha) * q_part * factor
            y_val[count] += math.cos(-element.alpha) * q_part * factor
        count += 1

    x_val = np.append(x_val, element.point_2.x)
    y_val = np.append(y_val, -element.point_2.z)
    x_val = np.insert(x_val, 0, element.point_1.x)
    y_val = np.insert(y_val, 0, -element.point_1.z)

    if element.q_load:
        sagging_moment = (-element.node_1.Ty + element.node_2.Ty) * 0.5 + 0.125 * element.q_load * element.l ** 2
        return x_val, y_val, sagging_moment
    else:
        return x_val, y_val

