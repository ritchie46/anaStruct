from anastruct.fem.system import SystemElements
from anastruct.vertex import vertex_range


def discretize(system, n=10):
    ss = SystemElements(EA=system.EA, EI=system.EI, load_factor=system.load_factor,
                        mesh=system.plotter.mesh, plot_backend=system.plotter.backend)

    for element in system.element_map.values():
        g = system.element_map[element.id].dead_load
        mp = system.non_linear_elements[element.id] if element.id in system.non_linear_elements else None

        for i, v in enumerate(vertex_range(element.vertex_1, element.vertex_2, n)):
            if i == 0:
                last_v = v
                continue

            loc = [last_v, v]
            ss.add_element(loc, EA=element.EA, EI=element.EI, g=g, mp=mp, spring=element.springs)
            last_v = v

    # supports
    ss.show_structure()
    for node, direction in zip(system.supports_roll, system.supports_roll_direction):
        ss.add_support_roll((node.id - 1) * n + 1, direction)
    for node in system.supports_fixed:
        ss.add_support_fixed((node.id - 1) * n + 1)
    for node in system.supports_hinged:
        ss.add_support_hinged((node.id - 1) * n + 1)
    for args in system.supports_spring_args:
        ss.add_support_spring((args[0] - 1) * n + 1, *args[1:])

    # loads
    for node_id, forces in system.loads_point.items():
        ss.point_load((node_id - 1) * n + 1, Fx=forces[0], Fz=forces[1] / system.orientation_cs)
    for node_id, forces in system.loads_moment.items():
        ss.moment_load((node_id - 1) * n + 1, forces)
    for element_id, forces in system.loads_q.items():
        ss.q_load(q=forces / system.orientation_cs / system.load_factor,
                  element_id=element_id,
                  direction=system.element_map[element_id].q_direction)

    return ss



