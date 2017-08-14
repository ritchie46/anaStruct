import math


class Vertex_xz:
    def __init__(self, x, z=None):
        if isinstance(x, (tuple, list)):
            self.x = x[0]
            self.z = x[1]
        elif type(x) is Vertex_xz:
            self.x = x.x
            self.z = x.z
        else:
            self.x = x
            self.z = z

    @property
    def y(self):
        return -self.z

    def modulus(self):
        return math.sqrt(self.x**2 + self.z**2)

    def unit(self):
        return 1 / self.modulus() * self

    def displace_polar(self, alpha, radius, inverse_z_axis=False):
        if inverse_z_axis:
            self.x += math.cos(alpha) * radius
            self.z -= math.sin(alpha) * radius
        else:
            self.x += math.cos(alpha) * radius
            self.z += math.sin(alpha) * radius

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = Vertex_xz(other)
        if isinstance(other, (float, int)):
            x = self.x + other
            z = self.z + other
        else:
            x = self.x + other.x
            z = self.z + other.z
        return Vertex_xz(x, z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is list:
            other = Vertex_xz(other)
        if isinstance(other, (float, int)):
            x = self.x - other
            z = self.z - other
        else:
            x = self.x - other.x
            z = self.z - other.z
        return Vertex_xz(x, z)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if type(other) is list:
            other = Vertex_xz(other)
        if isinstance(other, (float, int)):
            x = self.x * other
            z = self.z * other
        else:
            x = self.x * other.x
            z = self.z * other.z
        return Vertex_xz(x, z)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(other) is Vertex_xz:
            x = self.x / other.x
            z = self.z / other.z
            return Vertex_xz(x, z)
        else:
            x = self.x / other
            z = self.z / other
            return Vertex_xz(x, z)

    def __eq__(self, other):
        if self.x == other.x and self.z == other.z:
            return True
        else:
            return False

    def __str__(self):
        return "x: %s, z: %s" % (self.x, self.z)


class Vertex_xy(Vertex_xz):
    def __init__(self, x, y):
        """
        Vertex respecting the 'standard' x, y coordinate system.
        :param x: (flt)
        :param y: (flt)
        """
        Vertex_xz.__init__(self, x, -y)

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = Vertex_xz(other)
        if isinstance(other, (float, int)):
            x = self.x + other
            z = self.z - other
        else:
            x = self.x + other.x
            z = self.z - other.z
        return Vertex_xz(x, z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is list:
            other = Vertex_xz(other)
        if isinstance(other, (float, int)):
            x = self.x - other
            z = self.z + other
        else:
            x = self.x - other.x
            z = self.z + other.z
        return Vertex_xz(x, z)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __str__(self):
        return "x: %s, y: %s" % (self.x, -self.z)