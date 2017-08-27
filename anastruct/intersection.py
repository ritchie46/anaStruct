from anastruct.vertex import Vertex


def on_segment(p, q, r):
    """"
    Given three co-linear points p, q, r, the function checks if point q lies on line segment 'pr

    if q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y):
        return True
    """
    if min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.z, r.z) <= q.z <= max(p.z, r.z):
        return True
    else:
        return False


def orientation(p1, p2, p3):
    """
    :param p1: point object with x and z members
    :param p2: point object with x and z members
    :param p3: point object with x and z members
    :return: orientation

    To find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    See http://www.geeksforgeeks.org/orientation-3-ordered-points/
    """
    val = (p2.z - p1.z) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.z - p2.z)

    if val == 0:
        return 0  # co-linear
    if val > 0:
        return 1  # clockwise
    if val < 0:
        return 2  # counterclockwise


def do_intersect(line_1, line_2):
    """
    :param line_1: list with two coordinates [[x1, z1], [x2, z2]]
    :param line_2: list with two coordinates [[x1, z1], [x2, z2]]
    :return: boolean: True if intersect, False if not


    p1____________ q1

         q2
       /
      /
     /
    /
    p2
    """

    p1 = Vertex(x=line_1[0].x, z=line_1[0].z)
    q1 = Vertex(x=line_1[1].x, z=line_1[1].z)
    p2 = Vertex(x=line_2[0].x, z=line_2[0].z)
    q2 = Vertex(x=line_2[1].x, z=line_2[1].z)

    # check which orientations occur, clockwise, counterclockwise and co-linear
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # check for general case intersection
    if o1 != o2 and o3 != o4:
        return True

    # check for special cases
    # p1, q1 and p2 are co-linear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are co-linear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are co-linear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are co-linear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


