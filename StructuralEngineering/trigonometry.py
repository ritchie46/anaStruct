import math


class Pointxz:
    def __init__(self, x, z=None):
        if isinstance(x, (tuple, list)):
            self.x = x[0]
            self.z = x[1]
        elif type(x) is Pointxz:
            self.x = x.x
            self.z = x.z
        else:
            self.x = x
            self.z = z

    def modulus(self):
        return math.sqrt(self.x**2 + self.z**2)

    def displace_polar(self, alpha, radius, inverse_z_axis=False):
        if inverse_z_axis:
            self.x += math.cos(alpha) * radius
            self.z -= math.sin(alpha) * radius
        else:
            self.x += math.cos(alpha) * radius
            self.z += math.sin(alpha) * radius

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = Pointxz(other)
        if isinstance(other, (float, int)):
            x = self.x + other
            z = self.z + other
        else:
            x = self.x + other.x
            z = self.z + other.z
        return Pointxz(x, z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is list:
            other = Pointxz(other)
        if isinstance(other, (float, int)):
            x = self.x - other
            z = self.z - other
        else:
            x = self.x - other.x
            z = self.z - other.z
        return Pointxz(x, z)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if type(other) is Pointxz:
            x = self.x / other.x
            z = self.z / other.z
            return Pointxz(x, z)
        else:
            x = self.x / other
            z = self.z / other
            return Pointxz(x, z)

    def __eq__(self, other):
        if self.x == other.x and self.z == other.z:
            return True
        else:
            return False

    def __str__(self):
        return "x: %s, z: %s" % (self.x, self.z)


class Pointxy(Pointxz):
    def __init__(self, x, y):
        """
        Vertex respecting the 'standard' x, y coordinate system.
        :param x: (flt)
        :param y: (flt)
        """
        Pointxz.__init__(self, x, -y)

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = Pointxz(other)
        if isinstance(other, (float, int)):
            x = self.x + other
            z = self.z - other
        else:
            x = self.x + other.x
            z = self.z - other.z
        return Pointxz(x, z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is list:
            other = Pointxz(other)
        if isinstance(other, (float, int)):
            x = self.x - other
            z = self.z + other
        else:
            x = self.x - other.x
            z = self.z + other.z
        return Pointxz(x, z)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __str__(self):
        return "x: %s, y: %s" % (self.x, -self.z)