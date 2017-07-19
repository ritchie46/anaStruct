import unittest
from StructuralEngineering.FEM import system as se
import numpy as np


class SimpleTest(unittest.TestCase):

    def test_example_1(self):
        system = se.SystemElements()
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
        system = se.SystemElements()
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
        system = se.SystemElements()
        system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, K=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=10, element_id=2)
        sol = np.fromstring("""0.          0.          0.          0.06264607  0.00379285 -0.0128231
  0.0575402   0.01287382 -0.00216051  0.          0.         -0.0080909""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol, 1e-3))

    def test_example_4(self):
        system = se.SystemElements()
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000, hinge=2)
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring("""0.00000000e+00   0.00000000e+00  -3.25525753e-21   2.00000000e-09
   2.49096026e-25  -2.08333293e-03   0.00000000e+00   0.00000000e+00
   4.16666707e-03""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))

    def test_example_5(self):
        system = se.SystemElements()
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        sol = np.fromstring("""0.00000000e+00   0.00000000e+00   3.47221978e-04   2.66666645e-09
   6.66666453e-10  -6.94444356e-04   0.00000000e+00   0.00000000e+00
   3.47222298e-03""", float, sep=" ")
        self.assertTrue(np.allclose(system.solve(), sol))