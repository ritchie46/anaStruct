import unittest
from StructuralEngineering.FEM import system as se
import numpy as np
from StructuralEngineering.FEM.examples.ex_8_non_linear_portal import u4, ss as SS_8


class SimpleTest(unittest.TestCase):

    def test_example_1(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [3, -4]], EA=5e9, EI=8000)
        system.add_element(location_list=[[3, -4], [8, -4]], EA=5e9, EI=4000)
        system.q_load(element_id=2, q=10, direction=1)
        system.add_support_hinged(node_id=1)
        system.add_support_fixed(node_id=3)

        sol = np.fromstring("""0.00000000e+00   0.00000000e+00   1.30206878e-03   1.99999732e-08
           5.24999402e-08  -2.60416607e-03   0.00000000e+00   0.00000000e+00
           0.00000000e+00""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_2(self):
        system = se.SystemElements(xy_cs=False)
        system.add_truss_element(location_list=[[0, 0], [0, -5]], EA=5000)
        system.add_truss_element(location_list=[[0, -5], [5, -5]], EA=5000)
        system.add_truss_element(location_list=[[5, -5], [5, 0]], EA=5000)
        system.add_truss_element(location_list=[[0, 0], [5, -5]], EA=5000 * np.sqrt(2))
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=10, node_id=2)
        sol = np.fromstring("""0.00000000e+00   0.00000000e+00  -7.50739186e-03   4.00000000e-02
        2.44055338e-18  -4.95028396e-03   3.00000000e-02   1.00000000e-02
        -2.69147231e-03   0.00000000e+00   0.00000000e+00  -7.65426385e-03""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_3(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, k=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=10, element_id=2)
        sol = np.fromstring("""0.          0.          0.          0.06264607  0.00379285 -0.0128231
        0.0575402   0.01287382 -0.00216051  0.          0.         -0.0080909""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol, 1e-3))
        self.assertAlmostEqual(system.get_node_displacements(3)["ux"], 0.0575402011335)
        sol = system.get_node_results_system(3)
        self.assertAlmostEqual(sol["uz"], -0.012873819793265455)
        self.assertAlmostEqual(sol["phi_y"], 0.0021605118130397583)

    def test_example_4(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000, spring={2: 0})
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring("""0.00000000e+00   0.00000000e+00  -3.25525753e-21   2.00000000e-09
        2.49096026e-25  -2.08333293e-03   0.00000000e+00   0.00000000e+00
        4.16666707e-03""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_5(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring("""0.00000000e+00   0.00000000e+00   3.47221978e-04   2.66666645e-09
        6.66666453e-10  -6.94444356e-04   0.00000000e+00   0.00000000e+00
        3.47222298e-03""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_ex_6_fixed_hinge(self):
        """
        Test the primary force vector when applying a q_load at a hinged element.
        """

        ss = se.SystemElements()
        ss.add_element([[0, 0], [7, 0]], spring={2: 0})
        ss.add_element([7.1, 0])
        ss.add_support_fixed([1, 3])
        ss.q_load(10, 1)
        ss.solve()
        self.assertAlmostEqual(-61.25, ss.get_node_results_system(1)["Ty"], places=2)

    def test_ex_7_rotational_spring(self):
        """
        Test the rotational springs
        """
        from StructuralEngineering.FEM.examples.ex_7_rotational_spring import ss
        sol = np.fromstring("""0.          0.          0.          0.          0.23558645 -0.09665875
  0.          0.          0.06433688""", float, sep=" ")
        self.assertTrue(np.allclose(ss.solve(), sol))

    def test_ex_8(self):
        """
        Plastic hinges test
        """
        self.assertAlmostEqual(u4, 105.288412424)

    def test_find_node_id(self):
        self.assertEqual(SS_8.find_node_id([4, 4]), 6)
        self.assertEqual(SS_8.find_node_id([3, -3]), None)

if __name__ == "__main__":
    unittest.main()
