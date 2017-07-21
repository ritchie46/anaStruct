from StructuralEngineering.FEM.system import SystemElements, Pointxz
import math
EI = 5000

l = 3
fk = EI * math.pi**2 / ((2 * 3)**2)

print(fk)

p1 = Pointxz(0, 0)
p2 = p1 + [0,  -1]
p3 = p2 + [0, -1]
p4 = p3 + [1e-2, -1]

print(p3)
ss = SystemElements(EA=1e9, EI=EI, xy_cs=0)
ss.add_element([p1, p2])
ss.add_element([p2, p3])
ss.add_element([p3, p4])
ss.add_support_fixed(1)

# ss.show_structure()
ss.point_load(Fz=fk * 0.8, node_id=4)
ss.solve(gnl=1)

print(ss.element_map[3].node_2)
ss.show_displacement(factor=1)
