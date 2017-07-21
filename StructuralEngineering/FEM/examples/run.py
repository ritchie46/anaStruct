from StructuralEngineering.FEM import system as se
import numpy as np

system = se.SystemElements()
system.add_element(location_list=[[0, 0], [5, 0]], EA=5e9, EI=8000)
system.add_element(location_list=[[5, 0], [5, -5]], EA=5e9, EI=4000) #, mp={1: 8})
system.moment_load(Ty=100, node_id=3)
system.add_support_hinged(node_id=1)
system.add_support_hinged(node_id=3)

system.solve()

el = system.element_map[2]

system.show_displacement(show=0)