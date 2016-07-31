import math


class Point:
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def print_values(self):
        print(self.x, self.z)

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
        x = self.x + other.x
        z = self.z + other.z
        return Point(x, z)

    def __sub__(self, other):
        x = self.x - other.x
        z = self.z - other.z
        return Point(x, z)

    def __truediv__(self, other):
        if type(other) is Point:
            x = self.x / other.x
            z = self.z / other.z
            return Point(x, z)
        else:
            x = self.x / other
            z = self.z / other
            return Point(x, z)

    def __eq__(self, other):
        if self.x == other.x and self.z == other.z:
            return True
        else:
            return False
