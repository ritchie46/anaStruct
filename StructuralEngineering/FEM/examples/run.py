from StructuralEngineering.FEM.system import SystemElements, Pointxz


p1 = Pointxz(0, 0)
p2 = p1 + [1, 2]
p3 = p2 + [1, 1/3]
p4 = p3 + [1, 0]
p5 = p4 + [1, -1/3]
p6 = p5 + [1, -2]

points = [p1, p2, p3, p4, p5, p6]

ss = SystemElements(xy_cs=1)
for i in range(5):
    if i == 1:
        ss.add_element(points[i:i + 2], mp={2: 1})
    else:
        ss.add_element(points[i:i + 2])

    if i < 4:
        ss.point_load(Fz=10, Fx=10, node_id=i + 2)


ss.add_support_fixed(1)
ss.add_support_fixed(6)
# ss.show_structure()
ss.solve(gnl=1)
ss.show_bending_moment()
ss.show_displacement(factor=50)