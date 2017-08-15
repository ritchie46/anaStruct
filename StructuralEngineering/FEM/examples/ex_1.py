from StructuralEngineering.FEM import system as se


def run():
    system = se.SystemElements(xy_cs=False)
    system.add_element(location_list=[[0, 0], [3, -4]], EA=5e9, EI=8000)
    system.add_element(location_list=[[3, -4], [8, -4]], EA=5e9, EI=4000)

    system.q_load(element_id=2, q=10, direction=1)

    system.add_support_hinged(node_id=1)
    system.add_support_fixed(node_id=3)

    system.solve()
    system.show_structure()
    system.show_reaction_force()
    system.show_axial_force()
    system.show_shear_force()
    system.show_bending_moment()
    system.show_displacement()

if __name__ == "__main__":
    run()