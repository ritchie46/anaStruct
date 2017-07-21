import math
from StructuralEngineering.FEM import system as se


def run():
    system = se.SystemElements(invert_z=False)
    system.add_truss_element(location_list=[[0, 0], [0, -5]], EA=5000)
    system.add_truss_element(location_list=[[0, -5], [5, -5]], EA=5000)
    system.add_truss_element(location_list=[[5, -5], [5, 0]], EA=5000)
    system.add_truss_element(location_list=[[0, 0], [5, -5]], EA=5000 * math.sqrt(2))

    system.add_support_hinged(node_id=1)
    system.add_support_hinged(node_id=4)

    system.point_load(Fx=10, node_id=2)

    system.solve()
    system.show_structure()
    system.show_reaction_force()
    system.show_normal_force()
    system.show_shear_force()
    system.show_bending_moment()
    system.show_displacement()

if __name__ == "__main__":
    run()