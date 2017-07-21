from StructuralEngineering.FEM.system import SystemElements, Pointxz

p1 = Pointxz(0, 0)
p2 = p1 + [0.1,  3]
ss = SystemElements()
ss.add_element([p1, p2])
ss.add_support_fixed(1)
ss.point_load(Fz=1000, node_id=2)
ss.solve(gnl=0)

ss.show_displacement(factor=2)