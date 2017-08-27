from StructuralEngineering.FEM.system import SystemElements
from StructuralEngineering.material.profile import HEA
from StructuralEngineering.material.units import to_kNm2, to_kN
import time


load_factor = 3

E = 210e3
profile = HEA[180]
EA = to_kN(E * profile['A'])
EI = to_kNm2(E * profile['Iy'])
mp = profile["Wy"] * 235 * 1e-6




if __name__ == "__main__":
    fast = 1e5
    for i in range(10):
        t0 = time.time()

        ss = SystemElements(EA=EA, EI=EI, load_factor=load_factor)
        ss.add_element([[0, 0], [0, 4]], mp={2: mp})
        ss.add_element([0, 8], mp={1: mp, 2: mp})
        ss.add_element([2, 8], mp={1: mp, 2: mp})
        ss.add_element([4, 8], mp={1: mp, 2: mp})
        ss.add_element([4, 4], mp={1: mp, 2: mp})
        ss.add_element([4, 0], mp={1: mp, 2: mp})
        ss.add_truss_element([[0, 4], [4, 4]])
        ss.add_support_hinged(1)
        ss.add_support_fixed(7)

        ss.q_load(-20, 3)
        ss.q_load(-20, 4)
        ss.q_load(-1, 1)
        ss.q_load(-1, 2)
        ss.solve()

        fast = min(fast, time.time() - t0)

    print(fast)