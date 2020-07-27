from anastruct.fem.node import Node
from anastruct.vertex import Vertex
from anastruct.basic import FEMException, angle_x_axis

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements, Spring


def ensure_single_hinge(
    system: "SystemElements", spring: "Spring", node_id1: int, node_id2: int
):
    """
    Make sure that there is only a single hinge at a node.

    Parameters
    ----------
    node_id1
        Id of the node in the system
    node_id2
        Id of the node in the system
    """
    if spring is not None and 0 in spring:
        """
        Must be one rotational fixed element per node. Thus keep track of the hinges (k == 0).
        """

        for node_nr in range(1, 3):
            if spring[node_nr] == 0:  # node is a hinged node
                if node_nr == 1:
                    node_id = node_id1
                else:
                    node_id = node_id2

                system.node_map[node_id].hinge = True

                # check if more elements at this node are hinged
                if len(system.node_map[node_id].elements) > 0:

                    # TODO: Can be done better / simpler
                    # And should be tested
                    pass_hinge = not all(
                        [
                            node_id in el.hinges
                            for el in system.node_map[node_id].elements.values()
                        ]
                    )
                else:
                    pass_hinge = True
                if not pass_hinge:
                    del spring[node_nr]  # too many hinges at that element.


def append_node_id(
    system: "SystemElements",
    point_1: Vertex,
    point_2: Vertex,
    node_id1: int,
    node_id2: int,
):
    if node_id1 not in system.node_map:
        system.node_map[node_id1] = Node(node_id1, vertex=point_1)
    if node_id2 not in system.node_map:
        system.node_map[node_id2] = Node(node_id2, vertex=point_2)


def det_vertices(system, location_list):
    if isinstance(location_list, Vertex):
        point_1 = system._previous_point
        point_2 = Vertex(location_list)
    elif len(location_list) == 1:
        point_1 = system._previous_point
        point_2 = Vertex(location_list[0][0], location_list[0][1])
    elif isinstance(location_list[0], (int, float)):
        point_1 = system._previous_point
        point_2 = Vertex(location_list[0], location_list[1])
    elif isinstance(location_list[0], Vertex):
        point_1 = location_list[0]
        point_2 = location_list[1]
    else:
        point_1 = Vertex(location_list[0][0], location_list[0][1])
        point_2 = Vertex(location_list[1][0], location_list[1][1])
    system._previous_point = point_2

    return point_1, point_2


def det_node_ids(system, point_1, point_2):
    node_ids = []
    for p in (point_1, point_2):
        k = str(p)
        if k in system._vertices:
            node_id = system._vertices[k]
        else:
            node_id = len(system._vertices) + 1
            system._vertices[k] = node_id
        node_ids.append(node_id)
    return node_ids


def support_check(system, node_id):
    if system.node_map[node_id].hinge:
        raise FEMException(
            "Flawed inputs", "You cannot add a support to a hinged node."
        )


def force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp):
    """
    Forces the elements to be in the first and the last quadrant of the unity circle. Meaning the first node is
    always left and the last node is always right. Or the are both on one vertical line.

    The angle of the element will thus always be between -90 till +90 degrees.
    :return: point_1, point_2, node_id1, node_id2, spring, mp, ai
    """
    # determine the angle of the element with the global x-axis
    delta_x = point_2.x - point_1.x
    delta_z = point_2.z - point_1.z  # minus sign to work with an opposite z-axis
    angle = -angle_x_axis(delta_x, delta_z)

    if delta_x < 0:
        # switch points
        point_1, point_2 = point_2, point_1
        node_id1, node_id2 = node_id2, node_id1

        angle = -angle_x_axis(-delta_x, -delta_z)

        if spring is not None:
            assert type(spring) == dict, "The spring parameter should be a dictionary."
            if 1 in spring and 2 in spring:
                spring[1], spring[2] = spring[2], spring[1]
            elif 1 in spring:
                spring[2] = spring.pop(1)
            elif 2 in spring:
                spring[1] = spring.pop(2)

        if mp is not None:
            if 1 in mp and 2 in mp:
                mp[1], mp[2] = mp[2], mp[1]
            elif 1 in mp:
                mp[2] = mp.pop(1)
            elif 2 in mp:
                mp[1] = mp.pop(2)
    return point_1, point_2, node_id1, node_id2, spring, mp, angle
