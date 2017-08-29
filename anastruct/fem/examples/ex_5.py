from anastruct.fem import system as se


def run():
    system = se.SystemElements(xy_cs=False)
    system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000)
    system.add_element(location=[[5, 0], [5, -5]], EA=5e9, EI=4000)
    system.moment_load(Ty=10, node_id=3)
    system.add_support_hinged(node_id=1)
    system.add_support_hinged(node_id=3)

    system.solve()
    system.show_structure()
    system.show_reaction_force()
    system.show_axial_force()
    system.show_shear_force()
    system.show_bending_moment()
    system.show_displacement()

if __name__ == "__main__":
    run()