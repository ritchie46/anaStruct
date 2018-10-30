import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from anastruct.basic import find_nearest
try:
    from plotly.offline import plot_mpl, iplot_mpl
except Exception:
   import warnings
   warnings.warn("Plotly not found. The functions that require plotly will result in an error")
PATCH_SIZE = 0.03


class Plotter:
    def __init__(self, system, mesh, backend):
        self.system = system
        self.one_fig = None
        self.max_val = None
        self.max_force = 0
        self.max_q = 0

        self.max_system_point_load = 0
        self.mesh = max(3, mesh)
        self.backend = backend

    def __start_plot(self, figsize):
        plt.close("all")
        self.fig = plt.figure(figsize=figsize)
        self.one_fig = self.fig.add_subplot(111)
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
            factor = 0.15 * self.max_val / self.max_force
        return factor

    def __fixed_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """

        width = PATCH_SIZE * max_val
        height = PATCH_SIZE * max_val
        for node in self.system.supports_fixed:

            support_patch = mpatches.Rectangle((node.vertex.x - width * 0.5, - node.vertex.z - width * 0.5),
                                               width, height, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __hinged_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        if self.backend == "mpl":
            radius = PATCH_SIZE * max_val
            for node in self.system.supports_hinged:
                support_patch = mpatches.RegularPolygon((node.vertex.x, -node.vertex.z - radius),
                                                        numVertices=3, radius=radius, color='r', zorder=9)
                self.one_fig.add_patch(support_patch)

        else:  # backend is plotly
            x = []
            y = []
            s = 20 * max_val

            for node in self.system.supports_hinged:
                x.append(node.vertex.x)
                y.append(-node.vertex.z - s / 1000)
            self.one_fig.scatter(x, y, color='r', zorder=9, marker='^', s=s)

    def __roll_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = PATCH_SIZE * max_val
        count = 0
        for node in self.system.supports_roll:

            direction = self.system.supports_roll_direction[count]

            if direction == 2:  # horizontal roll
                support_patch = mpatches.RegularPolygon((node.vertex.x, -node.vertex.z - radius),
                                                        numVertices=3, radius=radius, color='r', zorder=9)
                self.one_fig.add_patch(support_patch)
                y = -node.vertex.z - 2 * radius
                self.one_fig.plot([node.vertex.x - radius, node.vertex.x + radius], [y, y], color='r')
            elif direction == 1:  # vertical roll
                center = 0
                x1 = center + math.cos(math.pi) * radius + node.vertex.x + radius
                z1 = center + math.sin(math.pi) * radius - node.vertex.z
                x2 = center + math.cos(math.radians(90)) * radius + node.vertex.x + radius
                z2 = center + math.sin(math.radians(90)) * radius - node.vertex.z
                x3 = center + math.cos(math.radians(270)) * radius + node.vertex.x + radius
                z3 = center + math.sin(math.radians(270)) * radius - node.vertex.z

                triangle = np.array([[x1, z1], [x2, z2], [x3, z3]])
                # translate the support to the node

                support_patch = mpatches.Polygon(triangle, color='r', zorder=9)
                self.one_fig.add_patch(support_patch)

                y = -node.vertex.z - radius
                self.one_fig.plot([node.vertex.x + radius * 1.5, node.vertex.x + radius * 1.5], [y, y + 2 * radius],
                                  color='r')
            count += 1

    def __rotating_spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = PATCH_SIZE * max_val

        for node in self.system.supports_spring_y:
            r = np.arange(0, radius, 0.001)
            theta = 25 * math.pi * r / (0.2 * max_val)
            x_val = []
            y_val = []

            count = 0
            for angle in theta:
                x = math.cos(angle) * r[count] + node.vertex.x
                y = math.sin(angle) * r[count] - radius - node.vertex.z
                x_val.append(x)
                y_val.append(y)
                count += 1

            self.one_fig.plot(x_val, y_val, color='r', zorder=9)

            # Triangle
            support_patch = mpatches.RegularPolygon((node.vertex.x, -node.vertex.z - radius * 3),
                                                    numVertices=3, radius=radius * 0.9, color='r', zorder=9)
            self.one_fig.add_patch(support_patch)

    def __spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        h = PATCH_SIZE * max_val
        left = -0.5 * h
        right = 0.5 * h
        dh = 0.2 * h

        # plotly
        x = []
        y = []
        s = 20 * max_val

        for node in self.system.supports_spring_z:
            yval = np.arange(0, -9, -1, dtype=float)
            yval = yval * dh
            xval = np.array([0, 0, left, right, left, right, left, 0, 0])

            yval = yval - node.vertex.z
            xval = xval + node.vertex.x

            self.one_fig.plot(xval, yval, color='r', zorder=10)

            if self.backend == "mpl":
                # Triangle
                support_patch = mpatches.RegularPolygon((node.vertex.x, -node.vertex.z - h * 2.6),
                                                        numVertices=3, radius=h * 0.9, color='r', zorder=10)
                self.one_fig.add_patch(support_patch)
            else:
                x.append(node.vertex.x)
                y.append(min(yval))

        if self.backend != "mpl":
            self.one_fig.scatter(x, y, color='r', zorder=9, marker='^', s=s)
            # reset
            x = []
            y = []

        for node in self.system.supports_spring_x:
            xval = np.arange(0, 9, 1, dtype=float)
            xval *= dh
            yval = np.array([0, 0, left, right, left, right, left, 0, 0])

            xval += node.vertex.x
            yval -= node.vertex.z
            self.one_fig.plot(xval, yval, color='r', zorder=10)

            if self.backend == "mpl":
                # Triangle
                support_patch = mpatches.RegularPolygon((node.vertex.x + h * 1.7, -node.vertex.z - h),
                                                        numVertices=3, radius=h * 0.9, color='r', zorder=10)
                self.one_fig.add_patch(support_patch)
            else:
                x.append(node.vertex.x + h * 1.7)
                y.append(-node.vertex.z - s / 2000)

            if self.backend != "mpl":
                verts = [(0, 0), (-1, -1), (1, -1)]
                self.one_fig.scatter(x, y, color='r', zorder=9, marker=verts, s=s)

    def __q_load_patch(self, max_val, verbosity):
        """
        :param max_val: max scale of the plot

        xn1;yn1  q-load   xn1;yn1
        -------------------
        |__________________|
        x1;y1  element    x2;y2
        """

        for q_id in self.system.loads_q.keys():
            el = self.system.element_map[q_id]
            if el.q_load > 0:
                direction = 1
            else:
                direction = -1

            h = 0.05 * max_val * abs(el.q_load) / self.max_q
            x1 = el.vertex_1.x
            y1 = -el.vertex_1.z
            x2 = el.vertex_2.x
            y2 = -el.vertex_2.z

            if el.q_direction == "y":
                ai = 0
            elif el.q_direction == "x":
                ai = 0.5 * math.pi
            else:
                ai = -el.ai

            # - value, because the positive z of the system is opposite of positive y of the plotter
            xn1 = x1 + math.sin(ai) * h * direction
            yn1 = y1 + math.cos(ai) * h * direction
            xn2 = x2 + math.sin(ai) * h * direction
            yn2 = y2 + math.cos(ai) * h * direction
            self.one_fig.plot([x1, xn1, xn2, x2], [y1, yn1, yn2, y2], color='g')

            if verbosity == 0:
                # arrow
                xa_1 = (x2 - x1) * 0.2 + x1 + math.sin(ai) * 0.8 * h * direction
                ya_1 = (y2 - y1) * 0.2 + y1 + math.cos(ai) * 0.8 * h * direction
                len_x = math.sin(ai - math.pi) * 0.6 * h * direction
                len_y = math.cos(ai - math.pi) * 0.6 * h * direction
                xt = xa_1 + math.sin(-el.ai) * 0.4 * h * direction
                yt = ya_1 + math.cos(-el.ai) * 0.4 * h * direction
                # fc = face color, ec = edge color
                self.one_fig.arrow(xa_1, ya_1, len_x, len_y, head_width=h*0.25, head_length=0.2*h, ec='g', fc='g')
                self.one_fig.text(xt, yt, "q=%d" % el.q_load, color='k', fontsize=9, zorder=10)

    @staticmethod
    def __arrow_patch_values(Fx, Fz, node, h):
        """
        :param Fx: (float)
        :param Fz: (float)
        :param node: (Node object)
        :param h: (float) Is a scale variable
        :return: Variables for the matplotlib plotter
        """

        F = (Fx**2 + Fz**2)**0.5
        len_x = Fx / F * h
        len_y = -Fz / F * h
        x = node.vertex.x - len_x * 1.2
        y = -node.vertex.z - len_y * 1.2

        return x, y, len_x, len_y, F

    def __point_load_patch(self, max_plot_range, verbosity=0):
        """
        :param max_plot_range: max scale of the plot
        """

        for k in self.system.loads_point:
            Fx, Fz = self.system.loads_point[k]
            F = (Fx**2 + Fz**2)**0.5
            node = self.system.node_map[k]
            h = 0.1 * max_plot_range * F / self.max_system_point_load
            x, y, len_x, len_y, F = self.__arrow_patch_values(Fx, Fz, node, h)

            self.one_fig.arrow(x, y, len_x, len_y, head_width=h*0.15, head_length=0.2*h, ec='b', fc='orange',
                               zorder=11)
            if verbosity == 0:
                self.one_fig.text(x, y, "F=%d" % F, color='k', fontsize=9, zorder=10)

    def __moment_load_patch(self, max_val):

        h = 0.2 * max_val
        for k, v in self.system.loads_moment.items():

            node = self.system.node_map[k]
            if v > 0:
                self.one_fig.plot(node.vertex.x, -node.vertex.z, marker=r'$\circlearrowleft$', ms=25,
                              color='orange')
            else:
                self.one_fig.plot(node.vertex.x, -node.vertex.z, marker=r'$\circlearrowright$', ms=25,
                              color='orange')
            self.one_fig.text(node.vertex.x + h * 0.2, -node.vertex.z + h * 0.2, "T=%d" % v, color='k',
                              fontsize=9, zorder=10)

    def plot_structure(self, figsize, verbosity, show=False, supports=True, scale=1, offset=(0, 0), gridplot=False):
        """
        :param show: (boolean) if True, plt.figure will plot.
        :param supports: (boolean) if True, supports are plotted.
        :return:
        """
        if not gridplot:
            self.__start_plot(figsize)
        max_x = 0
        max_z = 0
        min_x = 0
        min_z = 0
        for el in self.system.element_map.values():
            # plot structure
            axis_values = plot_values_element(el)
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.one_fig.plot(x_val, y_val, color='black', marker='s')

            # determine max values for scaling the plotter
            max_x = np.max(np.abs(x_val)) if np.max(np.abs(x_val)) > max_x else max_x
            max_z = np.max(np.abs(y_val)) if np.max([abs(x) for x in y_val]) > max_z else max_z

            min_x = np.min(np.abs(x_val)) if np.min(np.abs(x_val)) < min_x else min_x
            min_z = np.min(np.abs(y_val)) if np.min(np.abs(y_val)) < min_z else min_z

        center_x = (max_x - min_x) / 2 + min_x + -offset[0]
        center_z = (max_z - min_z) / 2 + min_x + -offset[1]
        max_plot_range = max(max_x, max_z)
        self.max_val = max_plot_range
        range = max_plot_range * scale
        plusxrange = center_x + range
        plusyrange = center_z + range * figsize[1] / figsize[0]
        minxrange = center_x - range
        minyrange = center_z - range * figsize[1] / figsize[0]

        self.one_fig.axis([minxrange, plusxrange, minyrange, plusyrange])

        if verbosity == 0:
            for el in self.system.element_map.values():

                axis_values = plot_values_element(el)
                x_val = axis_values[0]
                y_val = axis_values[1]

                # add node ID to plot
                range = max_plot_range * 0.015
                self.one_fig.text(x_val[0] + range, y_val[0] + range, '%d' % el.node_id1, color='g', fontsize=9, zorder=10)
                self.one_fig.text(x_val[-1] + range, y_val[-1] + range, '%d' % el.node_id2, color='g', fontsize=9,
                                  zorder=10)

                # add element ID to plot
                factor = 0.02 * self.max_val
                x_val = (x_val[0] + x_val[-1]) / 2 - math.sin(el.ai) * factor
                y_val = (y_val[0] + y_val[-1]) / 2 + math.cos(el.ai) * factor

                self.one_fig.text(x_val, y_val, str(el.id), color='r', fontsize=9, zorder=10)

        # add supports
        if supports:
            self.__fixed_support_patch(max_plot_range * scale)
            self.__hinged_support_patch(max_plot_range * scale)
            self.__roll_support_patch(max_plot_range * scale)
            self.__rotating_spring_support_patch(max_plot_range * scale)
            self.__spring_support_patch(max_plot_range * scale)

        # add_loads
        self.__q_load_patch(max_plot_range, verbosity)
        self.__point_load_patch(max_plot_range, verbosity)
        self.__moment_load_patch(max_plot_range)

        if show:
            self.plot()
        else:
            return self.fig

    def _add_node_values(self, x_val, y_val, value_1, value_2, digits):
        offset = self.max_val * 0.015

        # add value to plot
        self.one_fig.text(x_val[1] - offset, y_val[1] + offset, "%s" % round(value_1, digits),
                          fontsize=9, ha='center', va='center', )
        self.one_fig.text(x_val[-2] - offset, y_val[-2] + offset, "%s" % round(value_2, digits),
                          fontsize=9, ha='center', va='center', )

    def _add_element_values(self, x_val, y_val, value, index, digits=2):
        self.one_fig.text(x_val[index], y_val[index], "%s" % round(value, digits),
                          fontsize=9, ha='center', va='center', )

    def plot_result(self, axis_values, force_1=None, force_2=None, digits=2, node_results=True):
        # plot force
        x_val = axis_values[0]
        y_val = axis_values[1]
        self.one_fig.plot(x_val, y_val, color='b')

        if node_results:
            self._add_node_values(x_val, y_val, force_1, force_2, digits)

    def plot(self):
        if self.backend == "mpl":
            plt.show()
        elif self.backend == "ipt":
            iplot_mpl(self.fig)
        else:
            plot_mpl(self.fig)

    def axial_force(self, factor, figsize, verbosity, scale, offset, show, gridplot=False):
        self.max_force = 0
        self.plot_structure(figsize, 1, scale=scale, offset=offset, gridplot=gridplot)

        node_results = True if verbosity == 0 else False

        if factor is None:
            # determine max factor for scaling
            for el in self.system.element_map.values():
                factor = self.__set_factor(el.N_1, el.N_2)

        for el in self.system.element_map.values():
            if math.isclose(el.N_1, 0, rel_tol=1e-5, abs_tol=1e-9):
                pass
            else:
                axis_values = plot_values_axial_force(el, factor)
                N1 = el.N_1
                N2 = el.N_2

                self.plot_result(axis_values, N1, N2, node_results=node_results)

                point = (el.vertex_2 - el.vertex_1) / 2 + el.vertex_1
                if el.N_1 < 0:
                    point.displace_polar(alpha=el.ai + 0.5 * math.pi, radius=0.5 * el.N_1 * factor, inverse_z_axis=True)

                    if verbosity == 0:
                        self.one_fig.text(point.x, point.y, "-", ha='center', va='center',
                                          fontsize=20, color='b')
                if el.N_1 > 0:
                    point.displace_polar(alpha=el.ai + 0.5 * math.pi, radius=0.5 * el.N_1 * factor, inverse_z_axis=True)

                    if verbosity == 0:
                        self.one_fig.text(point.x, point.y, "+", ha='center', va='center',
                                          fontsize=14, color='b')

        if show:
            self.plot()
        else:
            return self.fig

    def bending_moment(self, factor, figsize, verbosity, scale, offset, show, gridplot=False):
        self.plot_structure(figsize, 1, scale=scale, offset=offset, gridplot=gridplot)
        self.max_force = 0
        con = len(self.system.element_map[1].bending_moment)

        # determine max factor for scaling
        if factor is None:
            factor = 0
            for el in self.system.element_map.values():
                if el.q_load:
                    m_sag = (el.node_1.Ty - el.node_2.Ty) * 0.5 - 1 / 8 * el.all_q_load * el.l**2
                    value_1 = max(abs(el.node_1.Ty), abs(m_sag))
                    value_2 = max(value_1, abs(el.node_2.Ty))
                    factor = self.__set_factor(value_1, value_2)
                else:
                    factor = self.__set_factor(el.node_1.Ty, el.node_2.Ty)

        # determine the axis values
        for el in self.system.element_map.values():
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and not el.all_q_load:
                # If True there is no bending moment, so no need for plotting.
                pass

            else:
                axis_values = plot_values_bending_moment(el, factor, con)
                if verbosity == 0:
                    node_results = True
                else:
                    node_results = False
                self.plot_result(axis_values, abs(el.node_1.Ty), abs(el.node_2.Ty), node_results=node_results)

                if el.all_q_load:
                    m_sag = min(el.bending_moment)
                    index = find_nearest(el.bending_moment, m_sag)[1]
                    offset = -self.max_val * 0.05

                    if verbosity == 0:
                        x = axis_values[0][index] + math.sin(-el.ai) * offset
                        y = axis_values[1][index] + math.cos(-el.ai) * offset
                        self.one_fig.text(x, y, "%s" % round(m_sag, 1),
                                          fontsize=9)
        if show:
            self.plot()
        else:
            return self.fig

    def shear_force(self, figsize, verbosity, scale, offset, show, gridplot=False):
        self.plot_structure(figsize, 1, scale=scale, offset=offset, gridplot=gridplot)
        self.max_force = 0

        # determine max factor for scaling
        for el in self.system.element_map.values():
            shear_1 = max(el.shear_force)
            shear_2 = min(el.shear_force)
            factor = self.__set_factor(shear_1, shear_2)

        for el in self.system.element_map.values():
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and el.q_load is None:
                # If True there is no bending moment, thus no shear force, so no need for plotting.
                pass
            else:
                axis_values = plot_values_shear_force(el, factor)
                shear_1 = axis_values[-2]
                shear_2 = axis_values[-1]

                if verbosity == 0:
                    node_results = True
                else:
                    node_results = False

                self.plot_result(axis_values, shear_1, shear_2, node_results=node_results)
        if show:
            self.plot()
        else:
            return self.fig

    def reaction_force(self, figsize, verbosity, scale, offset, show, gridplot=False):
        self.plot_structure(figsize, 1, supports=False, scale=scale, offset=offset, gridplot=gridplot)

        h = 0.2 * self.max_val
        max_force = 0

        for node in self.system.reaction_forces.values():
            max_force = abs(node.Fx) if abs(node.Fx) > max_force else max_force
            max_force = abs(node.Fz) if abs(node.Fz) > max_force else max_force

        for node in self.system.reaction_forces.values():
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

                if verbosity == 0:
                    self.one_fig.text(x, y, "R=%s" % round(node.Fx, 2), color='k', fontsize=9, zorder=10)

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

                if verbosity == 0:
                    self.one_fig.text(x, y, "R=%s" % round(node.Fz, 2), color='k', fontsize=9, zorder=10)

            if not math.isclose(node.Ty, 0, rel_tol=1e-5, abs_tol=1e-9):
                """
                '$...$': render the strings using mathtext
                """
                if node.Ty > 0:
                    self.one_fig.plot(node.vertex.x, -node.vertex.z, marker=r'$\circlearrowleft$', ms=25,
                                      color='orange')
                if node.Ty < 0:
                    self.one_fig.plot(node.vertex.x, -node.vertex.z, marker=r'$\circlearrowright$', ms=25,
                                      color='orange')

                if verbosity == 0:
                    self.one_fig.text(node.vertex.x + h * 0.2, -node.vertex.z + h * 0.2, "T=%s" % round(node.Ty, 2),
                                  color='k', fontsize=9, zorder=10)
        if show:
            self.plot()
        else:
            return self.fig

    def displacements(self, factor, figsize, verbosity, scale, offset, show, linear, gridplot=False):
        self.plot_structure(figsize, 1, scale=scale, offset=offset, gridplot=gridplot)
        self.max_force = 0
        ax_range = None
        # determine max factor for scaling
        if factor is None:
            for el in self.system.element_map.values():
                u_node = max(abs(el.node_1.ux), abs(el.node_1.uz))
                if el.type == "general":
                    factor = self.__set_factor(el.max_deflection, u_node)
                else:  # element is truss
                    factor = self.__set_factor(u_node, 0)
                ax_range = self.one_fig.get_xlim()

        for el in self.system.element_map.values():
            axis_values = plot_values_deflection(el, factor, ax_range, linear)
            self.plot_result(axis_values, node_results=False)

            if el.type == "general":
                # index of the max deflection
                index = np.argmax(np.abs(el.deflection))

                if verbosity == 0:
                    x = (el.node_1.ux + el.node_2.ux) / 2
                    y = (-el.node_1.uz + -el.node_2.uz) / 2

                    if index != 0 or index != el.deflection.size:
                        self._add_element_values(axis_values[0], axis_values[1],
                                                 el.deflection[index] + (x**2 + y**2)**0.5, index, 3)
        if show:
            self.plot()
        else:
            return self.fig

    def results_plot(self, figsize, verbosity, scale, offset, show):
        """
        Aggregate all the plots in one grid plot.

        :param figsize: (tpl)
        :param verbosity: (int)
        :param scale: (flt)
        :param offset: (tpl)
        :param show: (bool)
        :return: Figure or None
        """
        plt.close("all")
        self.fig = plt.figure(figsize=figsize)
        a = 320
        self.one_fig = self.fig.add_subplot(a + 1)
        plt.title("structure")
        self.plot_structure(figsize, verbosity, show=False, scale=scale, offset=offset, gridplot=True)
        self.one_fig = self.fig.add_subplot(a + 2)
        plt.title("bending moment")
        self.bending_moment(None, figsize, verbosity, scale, offset, False, True)
        self.one_fig = self.fig.add_subplot(a + 3)
        plt.title("shear force")
        self.shear_force(figsize, verbosity, scale, offset, False, True)
        self.one_fig = self.fig.add_subplot(a + 4)
        plt.title("axial force")
        self.axial_force(None, figsize, verbosity, scale, offset, False, True)
        self.one_fig = self.fig.add_subplot(a + 5)
        plt.title("displacements")
        self.displacements(None, figsize, verbosity, scale, offset, False, False, True)
        self.one_fig = self.fig.add_subplot(a + 6)
        plt.title("reaction force")
        self.reaction_force(figsize, verbosity, scale, offset, False, True)

        if show:
            self.plot()
        else:
            return self.fig




"""
### element functions ###
"""


def plot_values_element(element):
    x_val = [element.vertex_1.x, element.vertex_2.x]
    y_val = [-element.vertex_1.z, -element.vertex_2.z]
    return x_val, y_val


def plot_values_shear_force(element, factor=1):
    x1 = element.vertex_1.x
    y1 = -element.vertex_1.z
    x2 = element.vertex_2.x
    y2 = -element.vertex_2.z

    shear_1 = element.shear_force[0]
    shear_2 = element.shear_force[-1]

    # apply angle ai
    x_1 = x1 + shear_1 * math.sin(-element.ai) * factor
    y_1 = y1 + shear_1 * math.cos(-element.ai) * factor
    x_2 = x2 + shear_2 * math.sin(-element.ai) * factor
    y_2 = y2 + shear_2 * math.cos(-element.ai) * factor

    x_val = np.array([x1, x_1, x_2, x2])
    y_val = np.array([y1, y_1, y_2, y2])
    return x_val, y_val, shear_1, shear_2


def plot_values_axial_force(element, factor):
    x1 = element.vertex_1.x
    y1 = -element.vertex_1.z
    x2 = element.vertex_2.x
    y2 = -element.vertex_2.z

    N1 = element.N_1
    N2 = element.N_2

    x_1 = x1 + N1 * math.cos(0.5 * math.pi + element.ai) * factor
    y_1 = y1 + N1 * math.sin(0.5 * math.pi + element.ai) * factor
    x_2 = x2 + N2 * math.cos(0.5 * math.pi + element.ai) * factor
    y_2 = y2 + N2 * math.sin(0.5 * math.pi + element.ai) * factor

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
    x1 = element.vertex_1.x
    y1 = -element.vertex_1.z
    x2 = element.vertex_2.x
    y2 = -element.vertex_2.z

    # Determine forces for horizontal element ai = 0
    T_left = element.node_1.Ty
    T_right = -element.node_2.Ty

    # apply angle ai
    x1 += T_left * math.sin(-element.ai) * factor
    y1 += T_left * math.cos(-element.ai) * factor
    x2 += T_right * math.sin(-element.ai) * factor
    y2 += T_right * math.cos(-element.ai) * factor

    x_val = np.linspace(0, 1, con)
    y_val = np.empty(con)
    dx = x2 - x1
    dy = y2 - y1

    # determine moment for 0 < x < length of the element
    count = 0
    for i in x_val:
        x_val[count] = x1 + i * dx
        y_val[count] = y1 + i * dy

        if element.q_load or element.dead_load:
            q = element.all_q_load

            # if element.q_direction == "x":
            #     q += np.sign(math.sin(element.ai))

            x = i * element.l
            q_part = (-0.5 * -q * x**2 + 0.5 * -q * element.l * x)

            x_val[count] += math.sin(-element.ai) * q_part * factor
            y_val[count] += math.cos(-element.ai) * q_part * factor
        count += 1

    x_val = np.append(x_val, element.vertex_2.x)
    y_val = np.append(y_val, -element.vertex_2.z)
    x_val = np.insert(x_val, 0, element.vertex_1.x)
    y_val = np.insert(y_val, 0, -element.vertex_1.z)

    return x_val, y_val


def plot_values_deflection(element, factor, ax_range, linear=False):
    ux1 = element.node_1.ux * factor
    uz1 = -element.node_1.uz * factor
    ux2 = element.node_2.ux * factor
    uz2 = -element.node_2.uz * factor

    x1 = element.vertex_1.x + ux1
    y1 = -element.vertex_1.z + uz1
    x2 = element.vertex_2.x + ux2
    y2 = -element.vertex_2.z + uz2

    if element.type == "general" and not linear:
        n = len(element.deflection)
        x_val = np.linspace(x1, x2, n)
        y_val = np.linspace(y1, y2, n)

        x_val = x_val + element.deflection * math.sin(element.ai) * factor
        y_val = y_val + element.deflection * -math.cos(element.ai) * factor

    else:  # truss element has no bending
        x_val = np.array([x1, x2])
        y_val = np.array([y1, y2])

    # if ax_range is not None:
    #
    #     print(np.max(x_val), np.max(y_val), ax_range[1], factor)
    #     if factor > 1e-6:
    #         if np.max(x_val) > ax_range[1] \
    #                 or np.max(y_val) > ax_range[1]\
    #                 or np.min(x_val) < ax_range[0]  \
    #                 or np.min(y_val) < ax_range[0]:
    #             factor *= 0.1
    #             return plot_values_deflection(element, factor, ax_range, linear)
    return x_val, y_val

