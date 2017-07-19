import numpy as np
import math


def find_closest_index(array, searchFor):
    """
    Find the closest index in an ordered array
    :param array: array searched in
    :param searchFor: value that is searched
    :return: index of the closest match
    """
    if len(array) == 1:
        return 0

    # determine the direction of the iteration
    if array[1] < array[0] > searchFor:  # iteration from big to small

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return 0
        elif array[0] > searchFor and array[-1] > searchFor:
            return len(array) - 1

        for i in range(len(array)):
            if array[i] <= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                # search for the index with the smallest absolute difference
                if searchFor - valueAtIndex <= previousValue - searchFor:
                    return i
                else:
                    return i - 1

    else:
        # array[0] is smaller than searchFor
        # iteration from small to big

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return len(array) - 1
        elif array[0] > searchFor and array[-1] > searchFor:
            return 0

        for i in range(len(array)):
            if array[i] >= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                if valueAtIndex - searchFor <= searchFor - previousValue:
                    return i
                else:
                    return i - 1


def is_moving_towards(test_node, node_position, displacement):
    """
    Tests if a node is displacing towards or away from the node.
    :param test_node: Nodes that stands still (vector)
    :param node_position: Node of which the displacment tested (vector)
    :param displacement: (vector)
    :return: (Boolean)
    """
    to_node = test_node - node_position
    return np.dot(to_node, displacement) > 0


def find_nearest(array, value):
    """
    :param array: (numpy array object)
    :param value: (float) value searched for
    :return: (tuple) nearest value, index
    """
    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index = (np.abs(array-value)).argmin()
    return array[index], index


def converge(lhs, rhs, div=3):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    return (np.abs(rhs) / np.abs(lhs) - 1) / div + 1


class BaseVertex:
    """
    The vector modification done in some methods are done under the assumption that the vector
    is from the origin to the Point. [0, 0, 0] --> [x, y, z)

    Update:
    The modifications are done relative from the origin.
    """
    def __init__(self, x, y=None, z=None):
        """
        :param x: (float) Representing x coordinate. Alternative (list) [x, y, z]
        :param y: (float) Representing y coordinate.
        :param z: (float) Representing z coordinate.
        """
        if isinstance(x, (tuple, list)):
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        elif type(x) is BaseVertex:
            self.x = x.x
            self.y = x.y
            self.z = x.z
        else:
            self.x = x
            self.y = y
            self.z = z

    def modulus(self):
        """
        Returns the absolute value of the vector.
        :return: (float)
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        """
        Returns the dot product of two vectors.
        :param other: (BaseVertex)
        :return: (float)
        """
        if isinstance(other, (tuple, list)):
            other = BaseVertex(other)
        return dot_product(self, other)

    def cross(self, other):
        """
        Returns the cross product of two vectors.
        :param other: (BaseVertex)
        :return: (BaseVertex)
        """
        if isinstance(other, (tuple, list)):
            other = BaseVertex(other)
        v = cross_product(self, other)
        return BaseVertex(v[0], v[1], v[2])

    def angle(self, other):
        """
        Angle between two vectors starting from the origin.
        :param other: (BaseVertex)
        """
        if isinstance(other, (tuple, list)):
            other = BaseVertex(other)
        return math.acos(self.dot(other) / (self.modulus() * other.modulus()))

    def unit(self):
        """
        Converts the vector to a unit vector.
        :return: (BaseVertex)
        """
        return 1 / self.modulus() * self

    def rotate(self, angle, k):
        """
        Rotate point around origin.
        'Rodrigues' rotation formula'
        :param angle: (flt) (radians)
        :param k: (BaseVertex/ list) Unit vector describing an axis of rotation about which v rotates by an angle θ.
                  In other words, k is the normal vector determined by the cross product of v1 x v2.
        :return (BaseVertex)
        """
        if isinstance(k, list):
            k = BaseVertex(k[0], k[1], k[2])
        k = k.unit()
        return self * math.cos(angle) + k.cross(self) * math.sin(angle) + k * (k.dot(self)) * (1 - math.cos(angle))

    def mirror(self, ax, p):
        """
        Mirror the vertex.

        :param ax: (int) Mirror direction, 1, 2, or 3.
        :param p: (BaseVertex/ list) Point on the mirror axis
        :return: (BaseVertex)
        """
        v = BaseVertex(self) - p
        if ax == 1:
            k = BaseVertex(0, 1, 0)
            v = v.rotate(-0.5 * math.pi, k)
            v *= [1, 1, -1]
            v = v.rotate(0.5 * math.pi, k) + p
        if ax == 2:
            k = BaseVertex(1, 0, 0)
            v = v.rotate(-0.5 * math.pi, k)
            v *= [1, 1, -1]
            v = v.rotate(0.5 * math.pi, k) + p
        else:
            k = BaseVertex(0, 0, 1)
            v = v.rotate(-0.5 * math.pi, k)
            v *= [1, 1, -1]
            v = v.rotate(0.5 * math.pi, k) + p

        return v

    def unit_normal_vector(self, other):
        """
        Determines the cross product and converts the result to a unit vector.
        :param other: (BaseVertex)
        :return: (BaseVertex)
        """
        v = self.cross(other)
        return v.unit()

    def to_list(self):
        """
        Converts BaseVertex to [x, y, z] list.
        :return: (list)
        """
        return [self.x, self.y, self.z]

    def isclose(self, other):
        if isinstance(other, (tuple, list)):
            other = BaseVertex(other)
        close = True
        for i in range(3):
            if not math.isclose(self.x, other.x):
                close = False
        return close

    def __str__(self):
        return "x: %s, y: %s, z: %s" % (self.x, self.y, self.z)

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = BaseVertex(other)
        if isinstance(other, (float, int)):
            x = self.x + other
            y = self.y + other
            z = self.z + other
        else:
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
        return BaseVertex(x, y, z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is list:
            other = BaseVertex(other)
        if isinstance(other, (float, int)):
            x = self.x - other
            y = self.y - other
            z = self.z - other
        else:
            x = self.x - other.x
            y = self.y - other.y
            z = self.z - other.z
        return BaseVertex(x, y, z)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __eq__(self, other):
        if type(other) is list:
            other = BaseVertex(other)
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False

    def __mul__(self, other):
        if type(other) is list:
            other = BaseVertex(other)
        if isinstance(other, (float, int)):
            x = self.x * other
            y = self.y * other
            z = self.z * other
        else:
            x = self.x * other.x
            y = self.y * other.y
            z = self.z * other.z
        return BaseVertex(x, y, z)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(other) is list:
            other = BaseVertex(other)
        if isinstance(other, (float, int)):
            x = self.x / other
            y = self.y / other
            z = self.z / other
        else:
            x = self.x / other.x
            y = self.y / other.y
            z = self.z / other.z
        return BaseVertex(x, y, z)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, power, modulo=None):
        x = self.x**power
        y = self.y**power
        z = self.z**power
        return BaseVertex(x, y, z)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y
        if item == 2:
            return self.z


def dot_product(a, b):
    """
    Determines the dot product of two arrays.
    :param a: (list/ BaseVertex object)
    :param b: (list/ BaseVertex object)
    :return: (list)
    """
    if type(a) is list:
        return sum([i * j for i, j in zip(a, b)])
    elif isinstance(a, BaseVertex):
        return dot_product([a.x, a.y, a.z], [b.x, b.y, b.z])


def cross_product(a, b):
    """
    Determines the cross product of vectors with three values.
    :param a: (list/ BaseVertex object)
    :param b: (list/ BaseVertex object)
    :return: (list)
    """
    if type(a) is list:
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]
    if isinstance(a, BaseVertex):
        return cross_product([a.x, a.y, a.z], [b.x, b.y, b.z])


class Vertex(BaseVertex):
    def __int__(self, x, z):
        BaseVertex.__init__(x, 0, z)


def angle_x_axis(delta_x, delta_z):
    if math.isclose(delta_x, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is vertical
        if delta_z < 0:
            ai = 1.5 * math.pi
        else:
            ai = 0.5 * math.pi
    elif math.isclose(delta_z, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is horizontal
        if delta_x > 0:
            ai = 0
        else:
            ai = math.pi
    elif delta_x > 0 and delta_z > 0:  # quadrant 1 of unity circle
        ai = math.atan(abs(delta_z) / abs(delta_x))
    elif delta_x < 0 < delta_z:  # quadrant 2 of unity circle
        ai = 0.5 * math.pi + math.atan(abs(delta_x) / abs(delta_z))
    elif delta_x < 0 and delta_z < 0:  # quadrant 3 of unity circle
        ai = math.pi + math.atan(abs(delta_z) / abs(delta_x))
    elif delta_z < 0 < delta_x:  # quadrant 4 of unity circle
        ai = 1.5 * math.pi + math.atan(abs(delta_x) / abs(delta_z))
    else:
        raise ValueError("Can't determine the angle of the given element")

    return ai

