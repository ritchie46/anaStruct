import numpy as np
import math
from anastruct.basic import converge


def stiffness_adaptation(system, verbosity, max_iter):
    system.solve(True, naked=True)
    if verbosity == 0:
        print("Starting stiffness adaptation calculation.")

    # check validity
    assert all([mp > 0 for mpd in system.non_linear_elements.values() for mp in mpd]), \
        "Cannot solve for an mp = 0. If you want a hinge set the spring stiffness equal to 0."

    for c in range(max_iter):
        factors = []

        # update the elements stiffnesses
        for k, v in system.non_linear_elements.items():
            el = system.element_map[k]

            for node_no, mp in v.items():
                if node_no == 1:
                    # Fast Ty
                    m_e = el.element_force_vector[2] + el.element_primary_force_vector[2]
                else:
                    # Fast Ty
                    m_e = el.element_force_vector[5] + el.element_primary_force_vector[5]

                if abs(m_e) > mp:
                    el.nodes_plastic[node_no - 1] = True
                if el.nodes_plastic[node_no - 1]:
                    factor = converge(m_e, mp)
                    factors.append(factor)
                    el.update_stiffness(factor, node_no)

        if not np.allclose(factors, 1, 1e-3):
            system.solve(force_linear=True, naked=True)
        else:
            system.post_processor.node_results_elements()
            system.post_processor.node_results_system()
            system.post_processor.reaction_forces()
            system.post_processor.element_results()
            break

    if c == max_iter - 1:
        print("Couldn't solve the in the amount of iterations given.")
    elif verbosity == 0:
        print("Solved in {} iterations".format(c))
    return system.system_displacement_vector


def gnl(self, verbosity=0):
    if verbosity == 0:
        print("Starting geometrical non linear calculation")

    last_abs_u = np.abs(self.solve(force_linear=True))

    for c in range(200):
        current_abs_u = np.abs(self.solve(gnl=True))

        global_increase = np.sum(current_abs_u) / np.sum(last_abs_u)
        last_abs_u = current_abs_u

        if math.isclose(global_increase, 1.0):
            break

    if verbosity == 0:
        print("Solved in {} iterations".format(c))