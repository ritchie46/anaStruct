import numpy as np
from pytest import approx, raises
from anastruct import LoadCase, LoadCombination, SystemElements
from .utils import pspec_context
from .fixtures.e2e_fixtures import *

"""
NOTE: Several tests in this file validate that the correct numerical engineering results
occur in specific tested scenarios. These results are thereby validated against analytical
results for the same scenarios (which comes from engineering theory). Rather than comment
every single occurrence of every engineering variable, the following general definitions
apply to any tests in which these occur:
- L = total member length (typically in metres)
- l = length of individual member span
- w = peak distributed load magnitude (typically in kN/m)
- W = total distributed load (often w * l, typically in kN)
- P = concentrated / point load magnitude (typically in kN)
- a = location of concentrated / point load from left end
- b = location of concentrated / point load from right end of
- EI = bending stiffness (typically in kN*m^2)
- EA = axial stiffness (typically in kN*m/m)
"""


def describe_end_to_end_tests():
    @pspec_context("End-to-End Tests (i.e. original unittest tests)")
    def describe():
        pass

    def context_example_1():
        @pspec_context("Example 1")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
        system.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)
        system.q_load(element_id=2, q=-10)
        system.add_support_hinged(node_id=1)
        system.add_support_fixed(node_id=3)

        def it_results_in_correct_solution():
            sol = np.fromstring(
                """0.00000000e+00   0.00000000e+00   1.30206878e-03   1.99999732e-08
            5.24999402e-08  -2.60416607e-03   0.00000000e+00   0.00000000e+00
            0.00000000e+00""",
                float,
                sep=" ",
            )
            assert system.solve() == approx(sol)

    def context_example_2():
        @pspec_context("Example 2")
        def describe():
            pass

        system = SystemElements()
        system.add_truss_element(location=[[0, 0], [0, 5]], EA=5000)
        system.add_truss_element(location=[[0, 5], [5, 5]], EA=5000)
        system.add_truss_element(location=[[5, 5], [5, 0]], EA=5000)
        system.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * np.sqrt(2))
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=10, node_id=2)

        def it_results_in_correct_solution():
            sol = np.fromstring(
                """0.00000000e+00   0.00000000e+00  -7.50739186e-03   4.00000000e-02
            2.44055338e-18  -4.95028396e-03   3.00000000e-02   1.00000000e-02
            -2.69147231e-03   0.00000000e+00   0.00000000e+00  -7.65426385e-03""",
                float,
                sep=" ",
            )
            assert system.solve() == approx(sol)

    def context_example_3():
        @pspec_context("Example 3")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[[0, 0], [0, 5]], EA=15000, EI=5000)
        system.add_element(location=[[0, 5], [5, 5]], EA=15000, EI=5000)
        system.add_element(location=[[5, 5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, k=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=-10, element_id=2)
        system.solve()

        def it_results_in_correct_solution():
            sol = np.fromstring(
                """0.          0.          0.          0.06264607  0.00379285 -0.0128231
            0.0575402   0.01287382 -0.00216051  0.          0.         -0.0080909""",
                float,
                sep=" ",
            )
            assert system.solve() == approx(sol)

        def it_results_in_correct_node_3_displacements():
            assert system.get_node_displacements(3)["ux"] == approx(0.0575402011335)
            assert system.get_node_displacements(3)["uy"] == approx(-0.0128738197933)
            assert system.get_node_displacements(3)["phi_y"] == approx(0.0021605118130)

    def context_example_4():
        @pspec_context("Example 4")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000, spring={2: 0})
        system.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)

        def it_results_in_correct_solution():
            sol = np.fromstring(
                """0.00000000e+00  0.00000000e+00  1.46957616e-25  2.00000000e-09
            -7.34788079e-25  0.00000000e+00  0.00000000e+00  0.00000000e+00
            3.12500040e-03""",
                float,
                sep=" ",
            )
            assert system.solve() == approx(sol)

    def context_example_5():
        @pspec_context("Example 5")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)

        def it_results_in_correct_solution():
            sol = np.fromstring(
                """0.00000000e+00   0.00000000e+00   3.47221978e-04   -2.66666645e-09
            6.66666453e-10  -6.94444356e-04   0.00000000e+00   0.00000000e+00
            3.47222298e-03""",
                float,
                sep=" ",
            )
            assert system.solve() == approx(sol)

    def context_example_6_fixed_hinge():
        @pspec_context(
            "Example 6: Test the primary force vector when applying a q_load at a hinged element."
        )
        def describe():
            pass

        system = SystemElements()
        system.add_element([[0, 0], [7, 0]], spring={2: 0})
        system.add_element([7.1, 0])
        system.add_support_fixed([1, 3])
        system.q_load(-10, 1)
        system.solve()

        def it_results_in_correct_in_correct_moment_at_support():
            assert system.get_node_results_system(1)["Ty"] == approx(-61.25, rel=1e-3)

    def context_example_7_rotational_spring():
        @pspec_context("Example 7: Test the rotational springs")
        def describe():
            pass

        def it_results_in_correct_solution(SS_7):
            sol = np.fromstring(
                """0.          0.          0.          0.          0.23558645 -0.09665875
            0.          0.          0.06433688""",
                float,
                sep=" ",
            )
            assert np.allclose(SS_7.solve(), sol)

    def context_example_8_rotational_spring():
        @pspec_context("Example 8: Test the plastic hinges")
        def describe():
            pass

        def it_results_in_correct_total_displacement(SS_8):
            SS_8.solve()
            u4 = (
                SS_8.get_node_displacements(4)["uy"] ** 2
                + SS_8.get_node_displacements(4)["ux"] ** 2
            ) ** 0.5 * 1000
            assert u4 == approx(106.45829880642854)

    def context_example_11():
        @pspec_context("Example 11")
        def describe():
            pass

        def it_results_in_correct_axial_loads(SS_11):
            SS_11.solve()
            assert SS_11.element_map[1].N_1 == approx(27.8833333333)
            assert SS_11.element_map[1].N_2 == approx(17.8833333333)
            assert SS_11.get_element_results(1)["length"] == approx(5.3851647)

    def context_example_12():
        @pspec_context(
            "Example 12: Moment loads, with system nodes having 0 moment loads"
        )
        def describe():
            pass

        def it_results_in_correct_moment_loads(SS_12):
            SS_12.solve()
            assert SS_12.get_node_results_system(1)["Ty"] == approx(6.66667)
            assert SS_12.get_node_results_system(2)["Ty"] == approx(0)
            assert SS_12.get_node_results_system(3)["Ty"] == approx(0)
            assert SS_12.get_node_results_system(4)["Ty"] == approx(-6.66667)

    def context_example_13():
        @pspec_context("Example 13: X-axis loads, with system nodes having 0 Fx loads")
        def describe():
            pass

        def it_results_in_correct_moment_loads(SS_13):
            SS_13.solve()
            assert SS_13.get_node_results_system(1)["Fx"] == approx(6.66667)
            assert SS_13.get_node_results_system(2)["Fx"] == approx(0)
            assert SS_13.get_node_results_system(3)["Fx"] == approx(0)
            assert SS_13.get_node_results_system(4)["Fx"] == approx(-6.66667)

    def context_example_14():
        @pspec_context("Example 14: Tests dead load and parallel load on axis.")
        def describe():
            pass

        def it_results_in_correct_axial_loads(SS_14):
            SS_14.solve()
            assert SS_14.get_node_results_system(1)["Fx"] == approx(-5.18545)
            assert SS_14.get_node_results_system(2)["Fx"] == approx(0)
            assert SS_14.get_node_results_system(3)["Fx"] == approx(0)
            assert SS_14.get_node_results_system(4)["Fx"] == approx(0)
            assert SS_14.get_node_results_system(5)["Fx"] == approx(5.18545)

    def context_example_15():
        @pspec_context("Example 15: Tests dead load and parallel load on axis.")
        def describe():
            pass

        def it_results_in_correct_axial_loads(SS_15):
            SS_15.solve()
            assert SS_15.get_node_results_system(1)["Fx"] == approx(-9.41394)
            assert SS_15.get_node_results_system(2)["Fx"] == approx(0)
            assert SS_15.get_node_results_system(3)["Fx"] == approx(0)
            assert SS_15.get_node_results_system(4)["Fx"] == approx(0)
            assert SS_15.get_node_results_system(5)["Fx"] == approx(9.41394)

    def context_example_16():
        @pspec_context("Example 16: Tests parallel load on axis.")
        def describe():
            pass

        def it_results_in_correct_axial_loads(SS_16):
            SS_16.solve()
            assert SS_16.get_node_results_system(1)["Fx"] == approx(5.65685)
            assert SS_16.get_node_results_system(2)["Fx"] == approx(0)
            assert SS_16.get_node_results_system(3)["Fx"] == approx(0)
            assert SS_16.get_node_results_system(4)["Fx"] == approx(0)
            assert SS_16.get_node_results_system(5)["Fx"] == approx(5.65685)

    def context_example_18():
        @pspec_context("Example 18: Buckling factor")
        def describe():
            pass

        def it_results_in_correct_buckling_factor(SS_18):
            SS_18.solve(geometrical_non_linear=True)
            assert SS_18.buckling_factor == approx(600)

    def context_example_19():
        @pspec_context("Example 19: Numerical displacement averaging")
        def describe():
            pass

        def it_results_in_correct_deflections(SS_19):
            SS_19.solve()
            assert [el.deflection.max() for el in SS_19.element_map.values()] == approx(
                [0.10211865035814169, 0.10211865035814176]
            )

    def context_example_20():
        @pspec_context("Example 20: Inserting a node")
        def describe():
            pass

        def it_successfully_inserts_node(SS_20):
            x, y = SS_20.show_structure(values_only=True)
            assert x == approx(np.array([0.0, 3.0, 3.0, 5.0, 5.0, 10.0]))
            assert y == approx(np.array([0.0, 0.0, 0.0, 5.0, 5.0, 0.0]))

    def context_find_node_id():
        @pspec_context("find_node_id() function using Example 8")
        def describe():
            pass

        def it_finds_the_node_ids(SS_8):
            assert SS_8.find_node_id([4, 4]) == 6
            assert SS_8.find_node_id([3, -3]) is None

    def context_add_multiple_elements():
        @pspec_context("add_multiple_elements() function")
        def describe():
            pass

        system = SystemElements()
        system.add_multiple_elements([[0, 0], [10, 10]], n=5)

        def it_adds_multiple_elements():
            node_x_locations = [x.vertex.x for x in system.node_map.values()]
            assert node_x_locations == [0, 2.0, 4.0, 6.0, 8.0, 10]

    def context_no_forces_assertion():
        @pspec_context("Assertion error when no forces are entered")
        def describe():
            pass

        system = SystemElements()
        system.add_element([0, 10])

        def it_raises_an_error_with_no_forces():
            with raises(AssertionError):
                system.solve()

    def context_inclined_horizontal_rollers():
        @pspec_context("Inclined rollers are equal to horizontal rollers")
        def describe():
            pass

        system = SystemElements()
        x = [0, 1, 2]
        y = [0, 1, 0]
        system.add_element_grid(x, y)
        system.add_support_hinged(1)
        system.add_support_roll(3, "x")
        system.point_load(2, Fy=-100)
        system.solve()
        u1 = system.get_node_results_system(3)

        system = SystemElements()
        system.add_element_grid(x, y)
        system.add_support_hinged(1)
        system.add_support_roll(3, angle=0)
        system.point_load(2, Fy=-100)
        system.solve()
        u2 = system.get_node_results_system(3)

        def it_results_in_same_forces():
            assert u1["Fx"] == approx(u2["Fx"])
            assert u1["Fy"] == approx(u2["Fy"])
            assert u1["Ty"] == approx(u2["Ty"])

    def context_inclined_roller_reactions():
        @pspec_context("Inclined roller forces are calculated correctly")
        def describe():
            pass

        system = SystemElements()
        x = [0, 1, 2]
        y = [0, 0, 0]
        system.add_element_grid(x, y)
        system.add_support_hinged(1)
        system.add_support_roll(3, angle=45)
        system.point_load(2, Fy=-100)
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(3)["Fx"] == approx(50)
            assert system.get_node_results_system(3)["Fy"] == approx(-50)

    def context_inclined_roller_qload():
        @pspec_context("Inclined rollers and q-loads work together")
        def describe():
            pass

        system = SystemElements(EA=356000, EI=1330)
        system.add_element(location=[[0, 0], [10, 0]])
        system.add_support_hinged(node_id=1)
        system.add_support_roll(node_id=-1, angle=45)
        system.q_load(q=-1, element_id=1, direction="element")
        system.solve()

        def it_results_in_correct_reactions_forces():
            assert system.get_node_results_system(1)["Fx"] == approx(-5)
            assert system.get_node_results_system(1)["Fy"] == approx(-5)
            assert system.get_element_results(1)["Nmax"] == approx(-5)
            assert system.get_element_results(1)["Nmin"] == approx(-5)

    def context_deflection_averaging_complex():
        @pspec_context("Deflection averaging test using example 26")
        def describe():
            pass

        def it_results_in_correct_deflections(SS_26):
            SS_26.solve()
            assert [el.deflection.max() for el in SS_26.element_map.values()] == approx(
                [0.012466198717546239, 6.1000892498263e-07]
            )

    def context_single_hinges_in_trusses():
        @pspec_context("Reducing elements in trusses to single hinges only")
        def describe():
            pass

        system = SystemElements(EA=68300, EI=128, mesh=50)
        system.add_element(
            location=[[0.0, 0.0], [2.5, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        system.add_element(
            location=[[0.0, 0.0], [2.5, 2.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        system.add_element(
            location=[[2.5, 0.0], [5.0, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        system.add_element(
            location=[[2.5, 2.0], [2.5, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        system.add_element(
            location=[[2.5, 2.0], [5.0, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=0, Fy=-20.0, node_id=3)
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fx"] == approx(-12.5)
            assert system.get_node_results_system(1)["Fy"] == approx(-10)
            assert system.get_node_results_system(1)["Ty"] == approx(0)
            assert system.get_node_results_system(3)["Ty"] == approx(0)

    def context_multiple_elements_spacing():
        @pspec_context(
            "Tests bug fix for ensuring even spacing of multiple elements "
            + "regardless of any floating point roundoff"
        )
        def describe():
            pass

        system = SystemElements(EI=5e3, EA=1e5)
        system.add_multiple_elements([[0, 0], [10, 0]], 100)
        system.add_support_fixed(1)
        system.point_load(system.id_last_node, Fy=-10)
        system.solve()

        def it_results_in_valid_calculation():
            assert system.get_node_results_system(-1)["uy"] == approx(2 / 3)

    def context_vertical_spring():
        @pspec_context("Test addition of vertical spring supports")
        def describe():
            pass

        system = SystemElements(mesh=250)
        system.add_element(
            location=[(0.0, 0), (10.0, 0)], EA=356000.0, EI=1332.0000000000002
        )
        system.add_support_hinged(node_id=1)
        system.add_support_spring(node_id=2, translation=2, k=50, roll=False)
        system.q_load(q=-1.0, element_id=1, direction="y")
        system.solve()

        def it_results_in_correct_nodal_displacement():
            assert system.get_node_results_system(2)["uy"] == approx(0.1)

    def context_rotational_roller_support():
        @pspec_context("Test addition of roller supports with rotational restraint")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[(0, 0), (0, 1)])
        system.add_support_fixed(node_id=1)
        system.add_support_roll(node_id=2, direction="x", rotate=False)
        system.q_load(q=-1000, element_id=1, direction="x")
        system.point_load(node_id=2, Fy=-100)
        system.solve()

        def it_results_in_correct_displacements():
            assert system.get_node_results_system(2)["ux"] == approx(0.00833333)
            assert system.get_node_results_system(2)["uy"] == approx(0)
            assert system.get_node_results_system(2)["phi_y"] == approx(0)

        def it_results_in_correct_reaction():
            assert system.get_node_results_system(2)["Ty"] == approx(166.6667)

    def context_rotational_support():
        @pspec_context("Test addition of rotation-only supports")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[(0, 0), (1, 0)])
        system.add_support_fixed(node_id=1)
        system.add_support_rotational(node_id=2)
        system.q_load(q=-1000, element_id=1, direction="y")
        system.point_load(node_id=2, Fx=-100)
        system.solve()

        def it_results_in_correct_displacements():
            assert system.get_node_results_system(2)["ux"] == approx(0.00666667)
            assert system.get_node_results_system(2)["uy"] == approx(0.00833333)
            assert system.get_node_results_system(2)["phi_y"] == approx(0)

        def it_results_in_correct_reaction():
            assert system.get_node_results_system(2)["Ty"] == approx(-166.6667)

    def context_load_cases():
        @pspec_context("Test a trivial load case")
        def describe():
            pass

        system = SystemElements()
        system.add_truss_element(location=[[0, 0], [1000, 0]])
        system.add_truss_element(location=[[0, 0], [500, 500]])
        system.add_truss_element(location=[[500, 500], [1000, 0]])
        system.add_support_hinged(node_id=2)
        system.add_support_roll(node_id=1, direction="x", angle=None)
        lc_dead = LoadCase("dead")
        lc_dead.point_load(node_id=[1], Fx=10)
        combination = LoadCombination("ULS")
        combination.add_load_case(lc_dead, factor=1.4)
        results = combination.solve(system)

        def it_results_in_correct_reactions():
            assert results["dead"].get_node_results_system(1)["Fx"] == approx(0)
            assert results["dead"].get_node_results_system(2)["Fx"] == approx(14)

    def context_perpendicular_trapezoidal_load():
        @pspec_context("Test a complex-loaded system with trapezoidal loads")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[(0, 0), (1, 0)])
        system.add_element(location=[(1, 0), (2, 1.5)])
        system.add_support_fixed(node_id=1)
        system.add_support_fixed(node_id=2)
        system.q_load(q=(0.1, 1), element_id=2, direction="element")
        system.point_load(node_id=1, Fx=15)
        system.point_load(node_id=2, Fy=-5)
        system.moment_load(node_id=3, Ty=-7)
        system.solve()

        def it_has_correct_trapezoidal_load():
            assert system.get_element_results(2)["q"][0] == approx(-0.1)
            assert system.get_element_results(2)["q"][1] == approx(-1)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(0)
            assert system.get_node_results_system(1)["Fx"] == approx(15)
            assert system.get_node_results_system(2)["Fy"] == approx(-4.45)
            assert system.get_node_results_system(2)["Fx"] == approx(-0.825)
            assert system.get_node_results_system(1)["Ty"] == approx(0)
            assert system.get_node_results_system(2)["Ty"] == approx(-5.8625)

    def context_internal_hinge_symmetry():
        @pspec_context("Test bug fix for asymmetric results with internal hinges")
        def describe():
            pass

        system = SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], spring={2: 0})
        system.add_element(location=[[5, 0], [10, 0]])
        system.q_load(element_id=1, q=-9)
        system.q_load(element_id=2, q=-9)
        system.add_support_fixed(node_id=1)
        system.add_support_fixed(node_id=3)
        system.solve()

        def it_results_in_symmetric_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(
                system.get_node_results_system(3)["Fy"]
            )

        def it_results_in_symmetric_element_demands():
            assert system.get_element_results(1)["Mmax"] == approx(
                system.get_element_results(2)["Mmax"]
            )

    def context_q_and_q_perp_loads():
        @pspec_context("Test addition of simultaneous q and q_perp loads")
        def describe():
            pass

        system = SystemElements()
        system.add_element([2, 2])
        system.add_support_hinged(1)
        system.add_support_hinged(2)
        system.q_load([-2, -2], 1, "element", q_perp=[-1, -1])
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(-3)
            assert system.get_node_results_system(1)["Fx"] == approx(1)

        def it_results_in_correct_element_demands():
            assert system.get_element_results(1)["Nmax"] == approx(np.sqrt(2))
            assert system.get_element_results(1)["Qmax"] == approx(2 * np.sqrt(2))

    def context_parallel_trapezoidal_load():
        @pspec_context("Test additional of parallel trapezoidal loads")
        def describe():
            pass

        system = SystemElements()
        system.add_element([0, 1])
        system.add_element([0, 2])
        system.add_support_hinged(1)
        system.add_support_roll(3, "y")
        system.q_load([0, -0.5], 1, "y")
        system.q_load([-0.5, -1], 2, "y")
        system.solve()

        def it_results_in_correct_axial_demand_range():
            assert system.get_element_results(2)["Nmin"] == approx(-0.75)
            assert system.get_element_results(2)["Nmax"] == approx(0)

        def it_results_in_correct_reaction():
            assert system.get_node_results_system(1)["Fy"] == approx(-1)

    def context_load_case_example():
        @pspec_context(
            "Test documentation example of load cases from "
            + "https://anastruct.readthedocs.io/en/latest/loadcases.html"
        )
        def describe():
            pass

        system = SystemElements()
        height = 10

        x = np.cumsum([0, 4, 7, 7, 4])
        y = np.zeros(x.shape)
        x = np.append(x, x[::-1])
        y = np.append(y, y + height)

        system.add_element_grid(x, y)
        system.add_element([[0, 0], [0, height]])
        system.add_element([[4, 0], [4, height]])
        system.add_element([[11, 0], [11, height]])
        system.add_element([[18, 0], [18, height]])
        system.add_support_hinged([1, 5])

        lc_wind = LoadCase("wind")
        lc_wind.q_load(q=-1, element_id=[10, 11, 12, 13, 5])
        system.apply_load_case(lc_wind)
        system.remove_loads()

        lc_cables = LoadCase("cables")
        lc_cables.point_load(node_id=[2, 3, 4], Fy=-100)

        combination = LoadCombination("ULS")
        combination.add_load_case(lc_wind, 1.5)
        combination.add_load_case(lc_cables, factor=1.2)
        results = combination.solve(system)

        wind_Qmin = -12.5054911
        wind_Qmax = 2.4945089
        wind_Fx = 37.5
        wind_Fy = -17.0454545
        cables_Qmin = -38.2209899
        cables_Qmax = -38.2209899
        cables_Fx = 66.0413922
        cables_Fy = -180

        def it_results_in_correct_wind_demands_reactions():
            assert wind_Qmin, results["wind"].get_element_results(5)["Qmin"] == approx(
                wind_Qmin
            )
            assert wind_Qmax, results["wind"].get_element_results(5)["Qmax"] == approx(
                wind_Qmax
            )
            assert wind_Fx, results["wind"].get_node_results_system(5)["Fx"] == approx(
                wind_Fx
            )
            assert wind_Fy, results["wind"].get_node_results_system(5)["Fy"] == approx(
                wind_Fy
            )

        def it_results_in_correct_cables_demands_reactions():
            assert cables_Qmin, results["cables"].get_element_results(5)[
                "Qmin"
            ] == approx(cables_Qmin)
            assert cables_Qmax, results["cables"].get_element_results(5)[
                "Qmax"
            ] == approx(cables_Qmax)
            assert cables_Fx, results["cables"].get_node_results_system(5)[
                "Fx"
            ] == approx(cables_Fx)
            assert cables_Fy, results["cables"].get_node_results_system(5)[
                "Fy"
            ] == approx(cables_Fy)

        def it_results_in_correct_combination_demands_reactions():
            assert results["combination"].get_element_results(5)["Qmin"] == approx(
                wind_Qmin + cables_Qmin
            )
            assert results["combination"].get_element_results(5)["Qmax"] == approx(
                wind_Qmax + cables_Qmax
            )
            assert results["combination"].get_node_results_system(5)["Fx"] == approx(
                wind_Fx + cables_Fx
            )
            assert results["combination"].get_node_results_system(5)["Fy"] == approx(
                wind_Fy + cables_Fy
            )


def describe_analytical_validation_tests():
    @pspec_context("Validation tests of results, based upon analytical equations")
    def describe():
        pass

    def context_simply_supported_beam_validation():
        @pspec_context("Simply-supported beam with UDL validation")
        def describe():
            pass

        w = 1
        l = 2
        EI = 10000
        EA = 1000

        system = SystemElements(EI=EI, EA=EA, mesh=10000)
        system.add_element([[0, 0], [l, 0]])
        system.add_support_hinged([1, 2])
        system.q_load(w, element_id=1)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(1)["Mmax"] == approx(w * l**2 / 8)
            assert system.get_element_results(1)["Qmax"] == approx(w * l / 2)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(w * l / 2)
            assert system.get_node_results_system(2)["Fy"] == approx(w * l / 2)

        def it_results_in_correct_deflections():
            assert system.get_element_results(1)["wmax"] == approx(
                -5 * w * l**4 / (384 * EI)
            )


def describea_analytical_validation_tests():
    @pspec_context("Validation tests of results, based upon analytical equations")
    def describe():
        pass

    p = 80
    l = 5
    a = l / 3
    EI = 14000

    system = SystemElements(EI=EI, mesh=10000)
    system.add_element([[0, 0], [a, 0]])
    system.add_element([[a, 0], [2 * a, 0]])
    system.add_element([[2 * a, 0], [l, 0]])
    system.add_support_hinged([1, 4])
    system.point_load(node_id=2, Fy=p)
    system.point_load(node_id=3, Fy=p)
    system.solve()

    def it_results_in_correct_moment_shear():
        assert system.get_element_results(2)["Mmax"] == approx(p * a)
        assert system.get_element_results(1)["Qmax"] == approx(p)

    def it_results_in_correct_reactions():
        assert system.get_node_results_system(1)["Fy"] == approx(p)
        assert system.get_node_results_system(4)["Fy"] == approx(p)

    def it_results_in_correct_deflections():
        assert system.get_node_results_system(3)["uy"] + ...
        system.get_element_results(2)["wmax"] == approx(
            -((p * a) / (24 * EI)) * (3 * (l**2) - 4 * (a**2))
        )
