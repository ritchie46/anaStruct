import copy
import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from anastruct.basic import arg_to_list

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements


class LoadCase:
    """
    Group different loads in a load case
    """

    def __init__(self, name: str):
        """
        :param name: (str) Name of the load case
        """
        self.name: str = name
        self.spec: dict = dict()
        self.c: int = 0

    def q_load(
        self,
        q: Union[float, Sequence[float]],
        element_id: Union[int, Sequence[int]],
        direction: Union[str, Sequence[str]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y", "parallel"
        """
        self.c += 1
        self.spec[f"q_load-{self.c}"] = dict(
            q=q,
            element_id=element_id,
            direction=direction,
            rotation=rotation,
            q_perp=q_perp,
        )

    def point_load(
        self,
        node_id: Union[int, Sequence[int]],
        Fx: Union[float, Sequence[float]] = 0,
        Fy: Union[float, Sequence[float]] = 0,
        rotation: Union[float, Sequence[float]] = 0,
    ) -> None:
        """
        Apply a point load to a node.

        :param node_id: (int/ list) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fy: (flt/ list) Force in global x direction.
        :param rotation: (flt/ list) Rotate the force clockwise. Rotation is in degrees.
        """
        self.c += 1
        self.spec[f"point_load-{self.c}"] = dict(
            node_id=node_id, Fx=Fx, Fy=Fy, rotation=rotation
        )

    def moment_load(
        self, node_id: Union[int, Sequence[int]], Ty: Union[float, Sequence[float]]
    ) -> None:
        """
        Apply a moment on a node.

        :param node_id: (int/ list) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        self.c += 1
        self.spec[f"moment_load-{self.c}"] = dict(node_id=node_id, Ty=Ty)

    def dead_load(
        self, element_id: Union[int, Sequence[int]], g: Union[float, Sequence[float]]
    ) -> None:
        """
        Apply a dead load in kN/m on elements.

        :param element_id: (int/ list) representing the element ID
        :param g: (flt/ list) Weight per meter. [kN/m] / [N/m]
        """
        self.c += 1
        self.spec[f"dead_load-{self.c}"] = dict(element_id=element_id, g=g)

    def __str__(self) -> str:
        return f"Loadcase {self.name}:\n" + pprint.pformat(self.spec)


class LoadCombination:
    def __init__(self, name: str):
        self.name: str = name
        self.spec: dict = dict()

    def add_load_case(
        self,
        lc: Union[LoadCase, Sequence[LoadCase]],
        factor: Union[float, Sequence[float]],
    ) -> None:
        """
        Add a load case to the load combination.

        :param lc: (:class:`anastruct.fem.util.LoadCase`)
        :param factor: (flt) Multiply all the loads in this LoadCase with this factor.
        """
        if isinstance(lc, LoadCase):
            n = 1
        else:
            n = len(lc)
        lc = arg_to_list(lc, n)
        factor = arg_to_list(factor, n)
        for i, lci in enumerate(lc):
            self.spec[lci.name] = [lci, factor[i]]

    def solve(
        self,
        system: "SystemElements",
        force_linear: bool = False,
        verbosity: int = 0,
        max_iter: int = 200,
        geometrical_non_linear: bool = False,
        **kwargs: Any,
    ) -> Dict[str, "SystemElements"]:
        """
        Evaluate the Load Combination.

        :param system: (:class:`anastruct.fem.system.SystemElements`) Structure to apply loads on.
        :param force_linear: (bool) Force a linear calculation. Even when the system has non
                             linear nodes.
        :param verbosity: (int) 0: Log calculation outputs. 1: silence.
        :param max_iter: (int) Maximum allowed iterations.
        :param geometrical_non_linear: (bool) Calculate second order effects and determine the
                                       buckling factor.
        :return: (ResultObject)

        Development **kwargs:
            :param naked: (bool) Whether or not to run the solve function without doing
                          post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or
                                      increase the number of elements created that are used for
                                      determining the buckling_factor
        """

        results = {}
        for lc, factor in self.spec.values():
            ss = copy.deepcopy(system)

            ss.load_factor = factor
            ss.apply_load_case(lc)
            ss.solve(
                force_linear, verbosity, max_iter, geometrical_non_linear, **kwargs
            )
            results[lc.name] = ss

        ss_combination = copy.deepcopy(system)
        ss_combination.post_processor.node_results_system()
        for lc_ss in results.values():
            for k in ss_combination.element_map:
                ss_combination.element_map[k] = (
                    ss_combination.element_map[k] + lc_ss.element_map[k]
                )
            for k in ss_combination.node_map:
                ss_combination.node_map[k].add_results(lc_ss.node_map[k])
        ss_combination.post_processor.reaction_forces()

        results["combination"] = ss_combination
        return results
