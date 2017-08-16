from StructuralEngineering.FEM.system import SystemElements

"""
Test dead loads on the structure. TO DO! Distributed axial force
"""

ss = SystemElements()
# ss.add_element([10, 0])
# ss.add_support_fixed(1)
# ss.q_load(10, 1)
# ss.solve()
#
# print(ss.get_node_results_system())
# ss.show_bending_moment()

ss.add_element([5, 4], spring={2: 0})
ss.add_element([[0, 6], [5, 4]])
ss.add_element([7, 3.2])
ss.add_support_hinged([1, 3])

q = 5
ss.q_load([q, q], [2, 3], direction='y')
ss.solve()
ss.show_structure()
print(ss.get_element_results())
#ss.show_reaction_force()
ss.show_shear_force()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_bending_moment()