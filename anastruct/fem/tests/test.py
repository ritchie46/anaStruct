import sys
import unittest
import numpy as np
from anastruct.fem import system as se
from anastruct import LoadCase, LoadCombination, SystemElements
from anastruct.fem.examples.ex_8_non_linear_portal import ss as SS_8
from anastruct.fem.examples.ex_7_rotational_spring import ss as SS_7
from anastruct.fem.examples.ex_11 import ss as SS_11
from anastruct.fem.examples.ex_12 import ss as SS_12
from anastruct.fem.examples.ex_13 import ss as SS_13
from anastruct.fem.examples.ex_14 import ss as SS_14
from anastruct.fem.examples.ex_15 import ss as SS_15
from anastruct.fem.examples.ex_16 import ss as SS_16
from anastruct.fem.examples.ex_17_gnl import ss as SS_17
from anastruct.fem.examples.ex_18_discretize import ss as SS_18
from anastruct.fem.examples.ex_19_num_displacements import ss as SS_19
from anastruct.fem.examples.ex_20_insert_node import ss as SS_20
from anastruct.fem.examples.ex_26_deflection import ss as SS_26


class SimpleTest(unittest.TestCase):
    def test_example_1(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
        system.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)
        system.q_load(element_id=2, q=-10)
        system.add_support_hinged(node_id=1)
        system.add_support_fixed(node_id=3)

        sol = np.fromstring(
            """0.00000000e+00   0.00000000e+00   1.30206878e-03   1.99999732e-08
           5.24999402e-08  -2.60416607e-03   0.00000000e+00   0.00000000e+00
           0.00000000e+00""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_2(self):
        system = se.SystemElements()
        system.add_truss_element(location=[[0, 0], [0, 5]], EA=5000)
        system.add_truss_element(location=[[0, 5], [5, 5]], EA=5000)
        system.add_truss_element(location=[[5, 5], [5, 0]], EA=5000)
        system.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * np.sqrt(2))
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=10, node_id=2)
        sol = np.fromstring(
            """0.00000000e+00   0.00000000e+00  -7.50739186e-03   4.00000000e-02
        2.44055338e-18  -4.95028396e-03   3.00000000e-02   1.00000000e-02
        -2.69147231e-03   0.00000000e+00   0.00000000e+00  -7.65426385e-03""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_3(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [0, 5]], EA=15000, EI=5000)
        system.add_element(location=[[0, 5], [5, 5]], EA=15000, EI=5000)
        system.add_element(location=[[5, 5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, k=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=-10, element_id=2)
        sol = np.fromstring(
            """0.          0.          0.          0.06264607  0.00379285 -0.0128231
        0.0575402   0.01287382 -0.00216051  0.          0.         -0.0080909""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(system.solve(), sol, 1e-3))
        self.assertAlmostEqual(system.get_node_displacements(3)["ux"], 0.0575402011335)
        sol = system.get_node_results_system(3)
        self.assertAlmostEqual(sol["uy"], 0.012873819793265455)
        self.assertAlmostEqual(sol["phi_y"], 0.0021605118130397583)

    def test_example_4(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000, spring={2: 0})
        system.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring(
            """0.00000000e+00  0.00000000e+00  1.46957616e-25  2.00000000e-09
        -7.34788079e-25  0.00000000e+00  0.00000000e+00  0.00000000e+00
        3.12500040e-03""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_5(self):
        system = se.SystemElements()
        system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring(
            """0.00000000e+00   0.00000000e+00   3.47221978e-04   2.66666645e-09
        6.66666453e-10  -6.94444356e-04   0.00000000e+00   0.00000000e+00
        3.47222298e-03""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_ex_6_fixed_hinge(self):
        """
        Test the primary force vector when applying a q_load at a hinged element.
        """

        ss = se.SystemElements()
        ss.add_element([[0, 0], [7, 0]], spring={2: 0})
        ss.add_element([7.1, 0])
        ss.add_support_fixed([1, 3])
        ss.q_load(-10, 1)
        ss.solve()
        self.assertAlmostEqual(-61.25, ss.get_node_results_system(1)["Ty"], places=2)

    def test_ex_7_rotational_spring(self):
        """
        Test the rotational springs
        """

        sol = np.fromstring(
            """0.          0.          0.          0.          0.23558645 -0.09665875
        0.          0.          0.06433688""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(SS_7.solve(), sol))

    def test_ex_8(self):
        """
        Plastic hinges test
        """
        SS_8.solve()
        u4 = (
            SS_8.get_node_displacements(4)["uy"] ** 2
            + SS_8.get_node_displacements(4)["ux"] ** 2
        ) ** 0.5 * 1000
        self.assertAlmostEqual(u4, 106.45829880642854, places=5)

    def test_ex_11(self):

        SS_11.solve()
        el = SS_11.element_map[1]
        self.assertAlmostEqual(el.N_1, 27.8833333333, places=3)
        self.assertAlmostEqual(el.N_2, 17.8833333333, places=3)
        self.assertAlmostEqual(
            SS_11.get_element_results(1)["length"], 5.3851647, places=5
        )

    def test_ex_12(self):
        """
        System nodes Ty equal to 0
        """

        SS_12.solve()

        sol = [
            6.6666666666666679,
            3.5527136788005009e-15,
            -3.5527136788005009e-15,
            -6.6666666666666679,
        ]
        sssol = [a[3] for a in SS_12.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_13(self):
        """
        System nodes Fx equal to 0
        """
        SS_13.solve()

        sol = [
            6.6666666666666661,
            1.7763568394002505e-15,
            -8.8817841970012523e-16,
            -6.666666666666667,
        ]
        sssol = [a[1] for a in SS_13.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_14(self):
        """
        Tests dead load and parallel load on axis.
        """
        SS_14.solve()

        sol = [
            -5.1854495715665756,
            -2.6645352591003757e-15,
            -7.9936057773011271e-15,
            1.2434497875801753e-14,
            5.1854495715665738,
        ]
        sssol = [a[1] for a in SS_14.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_15(self):
        """
        Tests dead load and parallel load on axis.
        """
        SS_15.solve()

        sol = [
            -9.4139433815692541,
            -8.8817841970012523e-15,
            1.7763568394002505e-15,
            3.5527136788005009e-15,
            9.4139433815692577,
        ]
        sssol = [a[1] for a in SS_15.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_16(self):
        SS_16.solve()
        if sys.version_info[1] >= 6:
            sol = [
                -4.4408920985006262e-15,
                5.6568542494923886,
                -1.2434497875801753e-14,
                7.9936057773011271e-15,
                5.6568542494923797,
            ]
            sssol = [a[1] for a in SS_16.get_node_results_system()]
            self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_17_buckling_factor(self):
        SS_17.solve(geometrical_non_linear=True)
        self.assertNotAlmostEqual(493.48022005446785, SS_17.buckling_factor)

    def test_ex_18_buckling_factor(self):
        SS_18.solve(geometrical_non_linear=True)
        self.assertNotAlmostEqual(493.48022005446785, SS_18.buckling_factor)

    def test_ex_19_nummerical_displacements_averaging(self):
        SS_19.solve()
        self.assertTrue(
            np.allclose(
                [el.deflection.max() for el in SS_19.element_map.values()],
                [0.10211865035814169, 0.10211865035814176],
            )
        )

    def test_ex_20_insert_node(self):
        x, y = SS_20.show_structure(values_only=True)
        self.assertTrue(np.allclose(x, np.array([0.0, 3.0, 3.0, 5.0, 5.0, 10.0])))
        self.assertTrue(np.allclose(y, np.array([0.0, 0.0, 0.0, 5.0, 5.0, 0.0])))

    def test_find_node_id(self):
        self.assertEqual(SS_8.find_node_id([4, 4]), 6)
        self.assertEqual(SS_8.find_node_id([3, -3]), None)

    def test_add_multiple_elements(self):
        ss = se.SystemElements()
        ss.add_multiple_elements([[0, 0], [10, 10]], n=5)
        sol = [0, 2.0, 4.0, 6.0, 8.0, 10]
        self.assertTrue(
            all(
                [
                    np.isclose(a, b)
                    for a, b in zip(sol, [x.vertex.x for x in ss.node_map.values()])
                ]
            )
        )

    def test_no_forces_assertion(self):
        ss = se.SystemElements()
        ss.add_element([0, 10])
        self.assertRaises(AssertionError, ss.solve)

    def test_inclined_roll_equal_to_horizontal_roll(self):
        ss = se.SystemElements()
        x = [0, 1, 2]
        y = [0, 1, 0]
        ss.add_element_grid(x, y)
        ss.add_support_hinged(1)
        ss.add_support_roll(3, "x")
        ss.point_load(2, Fy=-100)
        ss.solve()
        u1 = ss.get_node_results_system(3)
        ss = se.SystemElements()
        ss.add_element_grid(x, y)
        ss.add_support_hinged(1)
        ss.add_support_roll(3, angle=0)
        ss.point_load(2, Fy=-100)
        ss.solve()
        u2 = ss.get_node_results_system(3)
        for k in u1:
            self.assertTrue(np.isclose(u1[k], u2[k]))

    def test_inclined_roll_force(self):
        ss = se.SystemElements()
        x = [0, 1, 2]
        y = [0, 0, 0]
        ss.add_element_grid(x, y)
        ss.add_support_hinged(1)
        ss.add_support_roll(3, angle=45)
        ss.point_load(2, Fy=-100)
        ss.solve()
        self.assertAlmostEqual(50, ss.get_node_results_system(3)["Fx"])

    def test_inclined_roll_and_qload(self):
        ss = se.SystemElements(EA=356000, EI=1330)
        ss.add_element(location=[[0, 0], [10, 0]])
        ss.add_support_hinged(node_id=1)
        ss.add_support_roll(node_id=-1, angle=45)
        ss.q_load(q=-1, element_id=1, direction="element")
        ss.solve()
        self.assertAlmostEqual(-5, ss.get_node_results_system(1)["Fx"])
        self.assertAlmostEqual(-5, ss.get_node_results_system(1)["Fy"])
        self.assertAlmostEqual(-5, ss.get_element_results(1)["Nmax"])
        self.assertAlmostEqual(-5, ss.get_element_results(1)["Nmin"])

    def test_deflection_averaging(self):
        SS_26.solve()
        self.assertTrue(
            np.allclose(
                [el.deflection.max() for el in SS_26.element_map.values()],
                [0.012466198717546239, 6.1000892498263e-07],
            )
        )

    def test_truss_single_hinge(self):
        ss = se.SystemElements(EA=68300, EI=128, mesh=50)
        ss.add_element(
            location=[[0.0, 0.0], [2.5, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        ss.add_element(
            location=[[0.0, 0.0], [2.5, 2.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        ss.add_element(
            location=[[2.5, 0.0], [5.0, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        ss.add_element(
            location=[[2.5, 2.0], [2.5, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        ss.add_element(
            location=[[2.5, 2.0], [5.0, 0.0]],
            g=0,
            spring={1: 0, 2: 0},
        )
        ss.add_support_hinged(node_id=1)
        ss.add_support_hinged(node_id=4)
        ss.point_load(Fx=0, Fy=-20.0, node_id=3)
        ss.solve()
        self.assertAlmostEqual(-12.5, ss.get_node_results_system(1)["Fx"])
        self.assertAlmostEqual(-10, ss.get_node_results_system(1)["Fy"])
        self.assertAlmostEqual(0, ss.get_node_results_system(1)["Ty"])
        self.assertAlmostEqual(0, ss.get_node_results_system(3)["Ty"])

    def test_multiple_elements_spacing(self):
        ss = se.SystemElements(EI=5e3, EA=1e5)
        ss.add_multiple_elements([[0, 0], [10, 0]], 100)
        ss.add_support_fixed(1)
        ss.point_load(ss.id_last_node, Fy=-10)
        ss.solve()
        self.assertAlmostEqual(2 / 3, ss.get_node_results_system(-1)["uy"])

    def test_vertical_spring(self):
        ss = se.SystemElements(mesh=250)
        ss.add_element(
            location=[(0.0, 0), (10.0, 0)], EA=356000.0, EI=1332.0000000000002
        )
        ss.add_support_hinged(node_id=1)
        ss.add_support_spring(node_id=2, translation=2, k=50, roll=False)
        ss.q_load(q=-1.0, element_id=1, direction="y")
        ss.solve()
        self.assertAlmostEqual(0.1, ss.get_node_results_system(2)["uy"])

    def test_rotational_roller_support(self):
        ss = se.SystemElements()
        ss.add_element(location=[(0, 0), (0, 1)])
        ss.add_support_fixed(node_id=1)
        ss.add_support_roll(node_id=2, direction="x", rotate=False)
        ss.q_load(q=-1000, element_id=1, direction="x")
        ss.point_load(node_id=2, Fy=-100)
        ss.solve()
        self.assertAlmostEqual(0.0083333, ss.get_node_results_system(2)["ux"])
        self.assertAlmostEqual(0.0, ss.get_node_results_system(2)["uy"])
        self.assertAlmostEqual(0.0, ss.get_node_results_system(2)["phi_y"])
        self.assertAlmostEqual(166.6667083, ss.get_node_results_system(2)["Ty"])

    def test_rotational_support(self):
        ss = se.SystemElements()
        ss.add_element(location=[(0, 0), (1, 0)])
        ss.add_support_fixed(node_id=1)
        ss.add_support_rotational(node_id=2)
        ss.q_load(q=-1000, element_id=1, direction="y")
        ss.point_load(node_id=2, Fx=-100)
        ss.solve()
        self.assertAlmostEqual(0.0066667, ss.get_node_results_system(2)["ux"])
        self.assertAlmostEqual(0.0083333, ss.get_node_results_system(2)["uy"])
        self.assertAlmostEqual(0.0, ss.get_node_results_system(2)["phi_y"])
        self.assertAlmostEqual(-166.6667083, ss.get_node_results_system(2)["Ty"])

    def test_load_cases(self):
        ss = se.SystemElements()
        ss.add_truss_element(location=[[0, 0], [1000, 0]])
        ss.add_truss_element(location=[[0, 0], [500, 500]])
        ss.add_truss_element(location=[[500, 500], [1000, 0]])
        ss.add_support_hinged(node_id=2)
        ss.add_support_roll(node_id=1, direction="x", angle=None)
        lc_dead = LoadCase("dead")
        lc_dead.point_load(node_id=[1], Fx=10)
        combination = LoadCombination("ULS")
        combination.add_load_case(lc_dead, factor=1.4)
        results = combination.solve(ss)
        self.assertAlmostEqual(0, results["dead"].get_node_results_system(1)["Fx"])
        self.assertAlmostEqual(14, results["dead"].get_node_results_system(2)["Fx"])

    def test_perpendicular_trapezoidal_load(self):
        ss = se.SystemElements()
        ss.add_element(location=[(0, 0), (1, 0)])
        ss.add_element(location=[(1, 0), (2, 1.5)])
        ss.add_support_fixed(node_id=1)
        ss.add_support_fixed(node_id=2)
        ss.q_load(q=(0.1, 1), element_id=2, direction="element")
        ss.point_load(node_id=1, Fx=15)
        ss.point_load(node_id=2, Fy=-5)
        ss.moment_load(node_id=3, Ty=-7)
        ss.solve()
        self.assertAlmostEqual(-0.1, ss.get_element_results(2)["q"][0])
        self.assertAlmostEqual(-1, ss.get_element_results(2)["q"][1])
        self.assertAlmostEqual(0, ss.get_node_results_system(1)["Fy"])
        self.assertAlmostEqual(15, ss.get_node_results_system(1)["Fx"])
        self.assertAlmostEqual(-4.45, ss.get_node_results_system(2)["Fy"])
        self.assertAlmostEqual(-0.825, ss.get_node_results_system(2)["Fx"])
        self.assertAlmostEqual(0, ss.get_node_results_system(1)["Ty"])
        self.assertAlmostEqual(-5.8625, ss.get_node_results_system(2)["Ty"])

    def test_internal_hinge_symmetry(self):
        ss = se.SystemElements()
        ss.add_element(location=[[0, 0], [5, 0]], spring={2: 0})
        ss.add_element(location=[[5, 0], [10, 0]])
        ss.q_load(element_id=1, q=-9)
        ss.q_load(element_id=2, q=-9)
        ss.add_support_fixed(node_id=1)
        ss.add_support_fixed(node_id=3)
        ss.solve()
        self.assertAlmostEqual(
            ss.get_node_results_system(1)["Fy"], ss.get_node_results_system(3)["Fy"]
        )
        self.assertAlmostEqual(
            ss.get_element_results(1)["Mmax"], ss.get_element_results(2)["Mmax"]
        )

    def test_q_and_q_perp(self):
        ss = se.SystemElements()
        ss.add_element([2, 2])
        ss.add_support_hinged(1)
        ss.add_support_hinged(2)
        ss.q_load([-2, -2], 1, "element", q_perp=[-1, -1])
        ss.solve()
        self.assertAlmostEqual(-3, ss.get_node_results_system(1)["Fy"], 6)
        self.assertAlmostEqual(1, ss.get_node_results_system(1)["Fx"], 6)
        self.assertAlmostEqual(np.sqrt(2), ss.get_element_results(1)["Nmax"])
        self.assertAlmostEqual(2 * np.sqrt(2), ss.get_element_results(1)["Qmax"])

    def test_parallel_trapezoidal_load(self):
        ss = se.SystemElements()
        ss.add_element([0, 1])
        ss.add_element([0, 2])
        ss.add_support_hinged(1)
        ss.add_support_roll(3, "y")
        ss.q_load([0, -0.5], 1, "y")
        ss.q_load([-0.5, -1], 2, "y")
        ss.solve()
        self.assertAlmostEqual(-0.75, ss.get_element_results(2)["Nmin"])
        self.assertAlmostEqual(0, ss.get_element_results(2)["Nmax"])
        self.assertAlmostEqual(-1, ss.get_node_results_system(1)["Fy"])

    def test_load_cases_example(self):
        # Example as in https://anastruct.readthedocs.io/en/latest/loadcases.html
        ss = SystemElements()
        height = 10

        x = np.cumsum([0, 4, 7, 7, 4])
        y = np.zeros(x.shape)
        x = np.append(x, x[::-1])
        y = np.append(y, y + height)

        ss.add_element_grid(x, y)
        ss.add_element([[0, 0], [0, height]])
        ss.add_element([[4, 0], [4, height]])
        ss.add_element([[11, 0], [11, height]])
        ss.add_element([[18, 0], [18, height]])
        ss.add_support_hinged([1, 5])

        lc_wind = LoadCase("wind")
        lc_wind.q_load(q=-1, element_id=[10, 11, 12, 13, 5])
        ss.apply_load_case(lc_wind)
        ss.remove_loads()

        lc_cables = LoadCase("cables")
        lc_cables.point_load(node_id=[2, 3, 4], Fy=-100)

        combination = LoadCombination("ULS")
        combination.add_load_case(lc_wind, 1.5)
        combination.add_load_case(lc_cables, factor=1.2)

        results = combination.solve(ss)
        wind_Qmin = -12.5054911
        wind_Qmax = 2.4945089
        wind_Fx = 37.5
        wind_Fy = -17.0454545
        cables_Qmin = -38.2209899
        cables_Qmax = -38.2209899
        cables_Fx = 66.0413922
        cables_Fy = -180
        self.assertAlmostEqual(
            wind_Qmin, results["wind"].get_element_results(5)["Qmin"]
        )
        self.assertAlmostEqual(
            wind_Qmax, results["wind"].get_element_results(5)["Qmax"]
        )
        self.assertAlmostEqual(
            wind_Fx, results["wind"].get_node_results_system(5)["Fx"]
        )
        self.assertAlmostEqual(
            wind_Fy, results["wind"].get_node_results_system(5)["Fy"]
        )
        self.assertAlmostEqual(
            cables_Qmin, results["cables"].get_element_results(5)["Qmin"]
        )
        self.assertAlmostEqual(
            cables_Qmax, results["cables"].get_element_results(5)["Qmax"]
        )
        self.assertAlmostEqual(
            cables_Fx, results["cables"].get_node_results_system(5)["Fx"]
        )
        self.assertAlmostEqual(
            cables_Fy, results["cables"].get_node_results_system(5)["Fy"]
        )
        self.assertAlmostEqual(
            wind_Qmin + cables_Qmin,
            results["combination"].get_element_results(5)["Qmin"],
        )
        self.assertAlmostEqual(
            wind_Qmax + cables_Qmax,
            results["combination"].get_element_results(5)["Qmax"],
        )
        self.assertAlmostEqual(
            wind_Fx + cables_Fx, results["combination"].get_node_results_system(5)["Fx"]
        )
        self.assertAlmostEqual(
            wind_Fy + cables_Fy, results["combination"].get_node_results_system(5)["Fy"]
        )


if __name__ == "__main__":
    unittest.main()
