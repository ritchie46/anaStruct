from anastruct.fem.system import SystemElements
"""
Test dead loads on the structure. TO DO! Distributed axial force
"""

q = 5
ss = SystemElements()
ss.add_element([[0, 6], [5, 4]], g=5)
ss.add_element([7, 3.2], g=5)
ss.add_element([[0, 0], [5, 4]], spring={2: 0})
ss.add_support_hinged([1, 4])
#ss.q_load([q, q], [1, 2], direction="y")

if __name__ == "__main__":
    ss.solve()
    print(ss.get_element_results())
    ss.show_structure()
    ss.show_axial_force()
