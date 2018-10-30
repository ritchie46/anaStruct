from anastruct.fem.system import SystemElements
from anastruct.fem.util.modify import discretize

ss = SystemElements(EI=5e3, EA=1e5)

ss.add_element([0, 10])

ss.add_support_hinged(1)
ss.add_support_roll(2, 1)

ss.point_load(2, Fz=-1)

ss = discretize(ss)
if __name__ == "__main__":
    ss.solve(geometrical_non_linear=True)
    print(ss.buckling_factor)
    ss.show_displacement()
