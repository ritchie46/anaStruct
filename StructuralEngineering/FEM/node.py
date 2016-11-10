class Node:
    def __init__(self, ID, Fx=0, Fz=0, Ty=0, ux=0, uz=0, phi_y=0, point=None):
        """
        :param ID: ID of the node, integer
        :param Fx: Value of Fx
        :param Fz: Value of Fz
        :param Ty: Value of Ty
        :param ux: Value of ux
        :param uz: Value of uz
        :param phi_y: Value of phi
        :param point: Point object
        """
        self.ID = ID
        # forces
        self.Fx = Fx
        self.Fz = Fz
        self.Ty = Ty
        # displacements
        self.ux = ux
        self.uz = uz
        self.phi_y = phi_y
        self.point = point
        self.hinge = False

    def show_result(self):
        print("\nID = %s\n"
              "Fx = %s\n"
              "Fz = %s\n"
              "Ty = %s\n"
              "ux = %s\n"
              "uz = %s\n"
              "phi_y = %s" % (self.ID, self.Fx, self.Fz, self.Ty, self.ux, self.uz, self.phi_y))

    def __add__(self, other):
        assert(self.ID == other.ID), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx + other.Fx
        Fz = self.Fz + other.Fz
        Ty = self.Ty + other.Ty
        ux = self.ux + other.ux
        uz = self.uz + other.uz
        phi_y = self.phi_y + other.phi_y

        return Node(self.ID, Fx, Fz, Ty, ux, uz, phi_y, self.point)

    def __sub__(self, other):
        assert (self.ID == other.ID), "Cannot subtract nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx - other.Fx
        Fz = self.Fz - other.Fz
        Ty = self.Ty - other.Ty
        ux = self.ux - other.ux
        uz = self.uz - other.uz
        phi_y = self.phi_y - other.phi_y

        return Node(self.ID, Fx, Fz, Ty, ux, uz, phi_y, self.point)


