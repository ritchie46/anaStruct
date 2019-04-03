import copy
from math import sqrt, sin, cos, atan, pi

"""
Code is extracted from DxfStructure project.
"""

checktolerance = 0.001

class Point():
    def __init__(self, xylist=[1.0, 2.0]):
        self.x = float(xylist[0])
        self.y = float(xylist[1])
    
    def distance(self, some_point):
        dist = sqrt((self.x - some_point.x)**2+(self.y - some_point.y)**2)
        return dist

class Line():
    def __init__(self, p1=Point([1.0, 1.0]), p2=Point([2.0, 3.0])):
        self.p1 = p1
        self.p2 = p2
        #---
        if self.p1.x == self.p2.x:
            self.A=1
            self.B=0
            self.C=-self.p1.x
        else:
            self.A = (self.p1.y - self.p2.y) / (self.p1.x - self.p2.x)
            self.B = -1.0
            self.C = self.p1.y - self.A * self.p1.x
            
    def distance(self, some_point):
        A = self.A
        B = self.B
        C = self.C
        #---
        dist_dl = abs(A*some_point.x + B*some_point.y + C)/sqrt((A)**2+(B)**2)
        dist_p1_p2 = self.p1.distance(self.p2)
        dist_o_p1 = self.p1.distance(some_point)
        dist_o_p2 = self.p2.distance(some_point)
        if (dist_p1_p2**2 + dist_o_p1**2 < dist_o_p2**2) or (dist_p1_p2**2 + dist_o_p2**2 < dist_o_p1**2):
            dist = min (dist_o_p1, dist_o_p2)
        else:
            dist = dist_dl
        return dist