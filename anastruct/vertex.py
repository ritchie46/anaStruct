import math
import numpy as np


class Vertex:
    def __init__(self, x, y=None):
        if isinstance(x, (tuple, list)):
            self.coordinates = np.array([x[0], x[1]])
        elif isinstance(x, np.ndarray):
            self.coordinates = x
        elif type(x) is Vertex:
            self.coordinates = np.array(x.coordinates)
        else:
            self.coordinates = np.array([x, y])

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[1] * -1

    def modulus(self):
        return np.sqrt(np.sum(self.coordinates**2))

    def unit(self):
        return 1 / self.modulus() * self

    def displace_polar(self, alpha, radius, inverse_z_axis=False):
        if inverse_z_axis:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] -= math.sin(alpha) * radius
        else:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] += math.sin(alpha) * radius

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates + other
        return Vertex(coordinates)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates - other
        return Vertex(coordinates)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates * other
        return Vertex(coordinates)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates / other
        return Vertex(coordinates)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "Vertex({}, {})".format(self.x, self.y)


def vertex_range(v1, v2, n):
    dv = v2 - v1
    return [v1 + dv * i / n for i in range(n + 1)]


