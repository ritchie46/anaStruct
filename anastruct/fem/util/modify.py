from anastruct.fem.system import SystemElements


def discretize(system):
    ss = SystemElements(EA=system.EA, EI=system.EI, load_factor=system.load_factor,
                        mesh=system.plotter.mesh, plot_backend=system.plotter.backend)

    for element in system.element_map.values():

        vertex_in_between = (element.vertex_2 - element.vertex_1) / 2 + element.vertex_1
        g = system.element_map[element.id].dead_load
        mp = system.non_linear_elements[element.id] if element.id in system.non_linear_elements else None

        loc_1 = [element.vertex_1, vertex_in_between]
        loc_2 = [vertex_in_between, element.vertex_2]

        ss.add_element(loc_1, EA=element.EA, EI=element.EI, g=g, mp=mp, spring=element.springs)
        ss.add_element(loc_2, EA=element.EA, EI=element.EI, g=g, mp=mp, spring=element.springs)

    # multiplier
    a = 2
    # supports
    for node, direction in zip(system.supports_roll, system.supports_roll_direction):
        ss.add_support_roll(node.id * a - 1, direction)
    for node in system.supports_fixed:
        ss.add_support_fixed(node.id * a - 1)
    for node in system.supports_hinged:
        ss.add_support_hinged(node.id * a - 1)
    for args in system.supports_spring_args:
        ss.add_support_spring(args[0] * a - 1, *args[1:])

    # loads
    for node_id, forces in system.loads_point.items():
        ss.point_load(node_id * a - 1, Fx=forces[0], Fz=forces[1] / system.orientation_cs)
    for node_id, forces in system.loads_moment.items():
        ss.moment_load(node_id * a - 1, forces)
    for element_id, forces in system.loads_q.items():
        ss.q_load(q=forces / system.orientation_cs / system.load_factor,
                  element_id=element_id,
                  direction=system.element_map[element_id].q_direction)

    return ss



