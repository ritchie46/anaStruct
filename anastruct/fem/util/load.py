import pprint
import copy
from anastruct.basic import args_to_lists
import numpy as np


class LoadCase:
    def __init__(self, name):
        self.name = name
        self.spec = dict()

    def q_load(self, q, element_id, direction="element"):
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"
        """
        self.spec['q_load'] = dict(q=q, element_id=element_id, direction=direction)

    def point_load(self, node_id, Fx=0, Fz=0, rotation=0):
        """
        Apply a point load to a node.

        :param node_id: (int/ list) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fz: (flt/ list) Force in global x direction.
        :param rotation: (flt/ list) Rotate the force clockwise. Rotation is in degrees.
        """
        self.spec['point_load'] = dict(node_id=node_id, Fx=Fx, Fz=Fz, rotation=rotation)

    def moment_load(self, node_id, Ty):
        """
        Apply a moment on a node.

        :param node_id: (int/ list) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        self.spec['moment_load'] = dict(node_id=node_id, Ty=Ty)

    def dead_load(self, element_id, g):
        """
        Apply a dead load in kN/m on elements.

        :param element_id: (int/ list) representing the element ID
        :param g: (flt/ list) Weight per meter. [kN/m] / [N/m]
        """
        self.spec['dead_load'] = dict(element_id=element_id, g=g)

    def __str__(self):
        return f'Loadcase {self.name}:\n' + pprint.pformat(self.spec)


class LoadCombination:
    def __init__(self, name):
        self.name = name
        self.spec = dict()

    def add_load_case(self, lc, factor):
        """
        Add a load case to the load combination.

        :param lc: (:class:`anastruct.fem.util.LoadCase`)
        :param factor: (flt) Multiply all the loads in this LoadCase with this factor.
        """
        lc, factor = args_to_lists(lc, factor)
        for i in range(len(lc)):
            self.spec[lc[i].name] = [lc[i], factor[i]]

    def solve(self, system, force_linear=False, verbosity=0, max_iter=200, geometrical_non_linear=False, **kwargs):
        """
        Evaluate the Load Combination.

        :param system: (:class:`anastruct.fem.system.SystemElements`) Structure to apply loads on.
        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0: Log calculation outputs. 1: silence.
        :param max_iter: (int) Maximum allowed iterations.
        :param geometrical_non_linear: (bool) Calculate second order effects and determine the buckling factor.
        :return: (ResultObject)

        Development **kwargs:
            :param naked: (bool) Whether or not to run the solve function without doing post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or increase the number
                                      of elements created that are used for determining the buckling_factor
        """
        results = {}
        for lc, factor in self.spec.values():
            ss = copy.deepcopy(system)
            ss.apply_load_case(lc)
            ss.solve(force_linear, verbosity, max_iter, geometrical_non_linear, **kwargs)
            results[lc.name] = ss

        ss_new = copy.deepcopy(system)

        for k in ss_new.element_map:
            for lc_ss in results.values():
                ss_new.element_map[k] = ss_new.element_map[k] + lc_ss.element_map[k]

        results['combination'] = ss_new
        return results
