from StructuralEngineering.FEM import system as se


def run():
    system = se.SystemElements()

    # Add beams to the system. Positive z-axis is down, positive x-axis is the right.
    system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
    system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
    system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)

    system.add_support_fixed(nodeID=1)
    # Add a rotational spring at node 4.
    system.add_support_spring(nodeID=4, translation=3, K=4000)

    system.point_load(Fx=30, nodeID=2)
    system.q_load(q=10, elementID=2)
    system.solve()
    system.show_structure()
    system.show_reaction_force()
    system.show_normal_force()
    system.show_shear_force()
    system.show_bending_moment()
    system.show_displacement()

if __name__ == "__main__":
    run()