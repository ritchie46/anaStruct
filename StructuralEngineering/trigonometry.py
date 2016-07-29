import math


class Point:
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def print_values(self):
        print(self.x, self.z)

    def modulus(self):
        return math.sqrt(self.x**2 + self.z**2)

    def subtract_point(self, point):
        self.x -= point.x
        self.z -= point.z
        return self

    def __eq__(self, other):
        if self.x == other.x and self.z == other.z:
            return True
        else:
            return False
