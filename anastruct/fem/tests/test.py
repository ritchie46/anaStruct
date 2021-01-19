import unittest
from anastruct.fem import system as se
from anastruct import LoadCase, LoadCombination
import numpy as np
from anastruct.fem.examples.ex_8_non_linear_portal import ss as SS_8
import sys


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
            """0.00000000e+00   0.00000000e+00  -3.25525753e-21   2.00000000e-09
        2.49096026e-25  -2.08333293e-03   0.00000000e+00   0.00000000e+00
        4.16666707e-03""",
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
        from anastruct.fem.examples.ex_7_rotational_spring import ss

        sol = np.fromstring(
            """0.          0.          0.          0.          0.23558645 -0.09665875
        0.          0.          0.06433688""",
            float,
            sep=" ",
        )
        self.assertTrue(np.allclose(ss.solve(), sol))

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
        from anastruct.fem.examples.ex_11 import ss

        ss.solve()
        el = ss.element_map[1]
        self.assertAlmostEqual(el.N_1, 27.8833333333, places=3)
        self.assertAlmostEqual(el.N_2, 17.8833333333, places=3)
        self.assertAlmostEqual(ss.get_element_results(1)["length"], 5.3851647, places=5)

    def test_ex_12(self):
        """
        System nodes Ty equal to 0
        """
        from anastruct.fem.examples.ex_12 import ss

        ss.solve()

        sol = [
            6.6666666666666679,
            3.5527136788005009e-15,
            -3.5527136788005009e-15,
            -6.6666666666666679,
        ]
        sssol = [a[3] for a in ss.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_13(self):
        """
        System nodes Fx equal to 0
        """
        from anastruct.fem.examples.ex_13 import ss

        ss.solve()

        sol = [
            6.6666666666666661,
            1.7763568394002505e-15,
            -8.8817841970012523e-16,
            -6.666666666666667,
        ]
        sssol = [a[1] for a in ss.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_14(self):
        """
        Tests dead load and parallel load on axis.
        """
        from anastruct.fem.examples.ex_14 import ss

        ss.solve()

        sol = [
            -5.1854495715665756,
            -2.6645352591003757e-15,
            -7.9936057773011271e-15,
            1.2434497875801753e-14,
            5.1854495715665738,
        ]
        sssol = [a[1] for a in ss.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_15(self):
        """
        Tests dead load and parallel load on axis.
        """
        from anastruct.fem.examples.ex_15 import ss

        ss.solve()

        sol = [
            -9.4139433815692541,
            -8.8817841970012523e-15,
            1.7763568394002505e-15,
            3.5527136788005009e-15,
            9.4139433815692577,
        ]
        sssol = [a[1] for a in ss.get_node_results_system()]
        self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_16(self):
        from anastruct.fem.examples.ex_16 import ss

        ss.solve()
        if sys.version_info[1] >= 6:
            sol = [
                -4.4408920985006262e-15,
                5.6568542494923886,
                -1.2434497875801753e-14,
                7.9936057773011271e-15,
                5.6568542494923797,
            ]
            sssol = [a[1] for a in ss.get_node_results_system()]
            self.assertTrue(all([np.isclose(a, b) for a, b in zip(sol, sssol)]))

    def test_ex_17_buckling_factor(self):
        from anastruct.fem.examples.ex_17_gnl import ss

        ss.solve(geometrical_non_linear=True)
        self.assertNotAlmostEqual(493.48022005446785, ss.buckling_factor)

    def test_ex_18_buckling_factor(self):
        from anastruct.fem.examples.ex_18_discretize import ss

        ss.solve(geometrical_non_linear=True)
        self.assertNotAlmostEqual(493.48022005446785, ss.buckling_factor)

    def test_ex_19_nummerical_displacements_averaging(self):
        from anastruct.fem.examples.ex_19_num_displacements import ss

        ss.solve()
        self.assertTrue(
            np.allclose(
                [el.deflection.max() for el in ss.element_map.values()],
                [0.10211865035814169, 0.10211865035814176],
            )
        )

    def test_ex_20_insert_node(self):
        from anastruct.fem.examples.ex_20_insert_node import ss

        x, y = ss.show_structure(values_only=True)
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
        self.assertAlmostEqual(-5, ss.get_element_results(1)["N"])

    def test_deflection_averaging(self):
        from anastruct.fem.examples.ex_26_deflection import ss

        ss.solve()
        self.assertTrue(
            np.allclose(
                [el.deflection.max() for el in ss.element_map.values()],
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


if __name__ == "__main__":
    unittest.main()
