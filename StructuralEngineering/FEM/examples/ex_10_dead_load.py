from StructuralEngineering.FEM.system import SystemElements

"""
Test dead loads on the structure. TO DO! Distributed axial force
"""

ss = SystemElements()
ss.add_element([[0, 0], [0, 14]], g=100)
ss.add_support_hinged(1)
ss.add_support_roll(2, 2)
print(ss.element_map[1].ai)
ss.solve()
print(ss.get_element_results())
#ss.show_reaction_force()
ss.show_axial_force()
ss.show_bending_moment(factor=0.001)