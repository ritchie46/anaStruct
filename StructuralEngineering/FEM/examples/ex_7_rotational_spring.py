from StructuralEngineering.FEM.system import SystemElements

"""
Test the primary force vector when applying a q_load at a hinged element.
"""

ss = SystemElements()
ss.add_element([[0, 0], [3.5, 0]])
ss.add_element([7, 0], spring={1: 100})
ss.add_support_fixed([1])
ss.add_support_hinged(3)
ss.point_load(2, Fz=100)

if __name__ == "__main__":
    print(str(ss.solve()))
    print(ss.get_node_results_system(2))
    print(ss.element_map[1].element_displacement_vector)
    print(ss.element_map[2].element_displacement_vector)
    ss.show_displacement()
    ss.show_reaction_force()
    ss.show_bending_moment()
