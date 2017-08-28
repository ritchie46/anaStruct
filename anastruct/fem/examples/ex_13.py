from anastruct.fem.system import SystemElements

ss = SystemElements()
ss.add_element([[0, 0], [2, 0]])
ss.add_element([4, 0])
ss.add_element([6, 0])
ss.add_support_fixed([1, 4])

ss.point_load([2, 3], [20, -20])

if __name__ == "__main__":
    ss.solve()
    print(ss.get_node_results_system())
    print([a[1] for a in ss.get_node_results_system()])
    ss.show_axial_force()
