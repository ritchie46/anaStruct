import unittest
from StructuralEngineering.FEM import system as se
import numpy as np


class SimpleTest(unittest.TestCase):

    def test_example_1(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [3, -4]], EA=5e9, EI=8000)
        system.add_element(location_list=[[3, -4], [8, -4]], EA=5e9, EI=4000)
        system.q_load(element_id=2, q=10, direction=1)
        system.add_support_hinged(node_id=1)
        system.add_support_fixed(node_id=3)
        system.solve()
        system.show_structure(show=False)
        system.show_bending_moment(show=False)
        system.show_normal_force(show=False)
        system.show_reaction_force(show=False)
        system.show_shear_force(show=False)

    def test_example_2(self):
        system = se.SystemElements(xy_cs=False)
        system.add_truss_element(location_list=[[0, 0], [0, -5]], EA=5000)
        system.add_truss_element(location_list=[[0, -5], [5, -5]], EA=5000)
        system.add_truss_element(location_list=[[5, -5], [5, 0]], EA=5000)
        system.add_truss_element(location_list=[[0, 0], [5, -5]], EA=5000 * np.sqrt(2))
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=4)
        system.point_load(Fx=10, node_id=2)
        system.solve()
        system.show_structure(show=False)
        system.show_bending_moment(show=False)
        system.show_normal_force(show=False)
        system.show_reaction_force(show=False)
        system.show_shear_force(show=False)

    def test_example_3(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
        system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)
        system.add_support_fixed(node_id=1)
        system.add_support_spring(node_id=4, translation=3, K=4000)
        system.point_load(Fx=30, node_id=2)
        system.q_load(q=10, element_id=2)
        system.solve()
        system.show_structure(show=False)
        system.show_bending_moment(show=False)
        system.show_normal_force(show=False)
        system.show_reaction_force(show=False)
        system.show_shear_force(show=False)

    def test_example_4(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000, hinge=2)
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        system.show_structure(show=False)
        system.solve()
        system.show_bending_moment(show=False)
        system.show_normal_force(show=False)
        system.show_reaction_force(show=False)
        system.show_shear_force(show=False)

    def test_example_5(self):
        system = se.SystemElements(xy_cs=False)
        system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000)
        system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000)
        system.moment_load(Ty=10, node_id=3)
        system.add_support_hinged(node_id=1)
        system.add_support_hinged(node_id=3)
        system.solve()
        system.show_structure(show=False)
        system.show_bending_moment(show=False)
        system.show_normal_force(show=False)
        system.show_reaction_force(show=False)
        system.show_shear_force(show=False)



if __name__ == "__main__":
    unittest.main()
