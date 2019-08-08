class Node:
    def __init__(self, id, Fx=0, Fz=0, Ty=0, ux=0, uz=0, phi_y=0, vertex=None):
        """
        :param id: ID of the node, integer
        :param Fx: Value of Fx
        :param Fz: Value of Fz
        :param Ty: Value of Ty
        :param ux: Value of ux
        :param uz: Value of uz
        :param phi_y: Value of phi
        :param vertex: Point object
        """
        self.id = id
        # forces
        self.Fx = Fx
        self.Fz = Fz
        self.Ty = Ty
        # displacements
        self.ux = ux
        self.uz = uz
        self.phi_y = phi_y
        self.vertex = vertex
        self.hinge = False
        self.elements = {}

    @property
    def Fy(self):
        return -self.Fz

    def __str__(self):
        if self.vertex:
            return "[id = {}, Fx = {}, Fz = {}, Ty = {}, ux = {}, uz = {}, phi_y = {}, x = {}, y = {}]".format(
                self.id,
                self.Fx,
                self.Fz,
                self.Ty,
                self.ux,
                self.uz,
                self.phi_y,
                self.vertex.x,
                self.vertex.y,
            )
        else:
            return "[id = {}, Fx = {}, Fz = {}, Ty = {}, ux = {}, uz = {}, phi_y = {}]".format(
                self.id, self.Fx, self.Fz, self.Ty, self.ux, self.uz, self.phi_y
            )

    def __add__(self, other):
        assert (
            self.id == other.id
        ), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx + other.Fx
        Fz = self.Fz + other.Fz
        Ty = self.Ty + other.Ty

        return Node(self.id, Fx, Fz, Ty, self.ux, self.uz, self.phi_y, self.vertex)

    def __sub__(self, other):
        assert (
            self.id == other.id
        ), "Cannot subtract nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx - other.Fx
        Fz = self.Fz - other.Fz
        Ty = self.Ty - other.Ty

        return Node(self.id, Fx, Fz, Ty, self.ux, self.uz, self.phi_y, self.vertex)

    def reset(self):
        self.Fx = self.Fz = self.Ty = self.ux = self.uz = self.phi_y = 0
