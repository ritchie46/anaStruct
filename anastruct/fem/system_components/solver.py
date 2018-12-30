import numpy as np
import copy
from anastruct.basic import converge
import logging
from scipy import linalg


def stiffness_adaptation(system, verbosity, max_iter):
    """
    Non linear solver for the nodes by adapting the stiffness of the elements (nodes).

    :param system: (SystemElements)
    :param verbosity: (int)
    :param max_iter: (int)
    :return: (np.array) Vector with displacements.
    """
    system.solve(True, naked=True)
    if verbosity == 0:
        logging.info("Starting stiffness adaptation calculation.")

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
        logging.warning("Couldn't solve the in the amount of iterations given. max_iter={}".format(max_iter))
    elif verbosity == 0:
        logging.info("Solved in {} iterations".format(c))
    return system.system_displacement_vector


def det_linear_buckling(system):
    """
    Determine linear buckling by solving the generalized eigenvalue problem (k -λkg)x = 0.

    geometrical stiffness matrix at buckling point: Kg = f(N_max)
    1st order forces: N0
    Nmax = λN0
    Kg(Nmax) = λ(Kg(N0) = λKg0

    2nd order analysis is solved by:
    (K + λKg0)U = F

    We are interested in the point that there is nog additional load F and displacement U is possible.
    (K + λKg0)ΔU = ΔF = 0
    (K + λKg0) = 0

    Is the generalized eigenvalue problem:
    (A - λB)x = 0

    :param system: (SystemElements)
    :return: (flt) The factor the loads can be increased until the structure fails due to buckling.
    """
    system.solve()

    # buckling
    k0 = system.reduced_system_matrix * 1.0  # copy

    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()
        el.reset()

    system.solve()
    kg = system.reduced_system_matrix - k0
    # solve (k -λkg)x = 0

    eigenvalues = np.abs(linalg.eigvals(k0, kg))
    return np.min(eigenvalues)


def geometrically_non_linear(system, verbosity=0, buckling_factor=True, discretize_kwargs=None):
    """

    :param system: (SystemElements)
    :param verbosity: (int)
    :param buckling_factor: (bool)
    :param discretize_kwargs: (dict) Containing the kwargs passed to the discretize function
    :param discretize: (function) discretize function.
    :return: buckling_factor: (flt) The factor the loads can be increased until the structure fails due to buckling.
    """
    # https://www.ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Lecture_2b.pdf
    if verbosity == 0:
        logging.info("Starting geometrical non linear calculation")

    if buckling_factor:
        buckling_system = copy.copy(system)
        if discretize_kwargs is not None:
            buckling_system.discretize(**discretize_kwargs)

        buckling_factor = det_linear_buckling(buckling_system)
    else:
        buckling_factor = None

    system.solve()

    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()

    system.solve()

    return buckling_factor



