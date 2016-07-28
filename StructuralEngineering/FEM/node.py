class Node:
    def __init__(self, ID, Fx=0, Fz=0, Ty=0, ux=0, uz=0, phiy=0, point=None):
        """
        :param ID: ID of the node, integer
        :param Fx: Value of Fx
        :param Fz: Value of Fz
        :param Ty: Value of Ty
        :param ux: Value of ux
        :param uz: Value of uz
        :param phiy: Value of phi
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
        self.phiy = phiy
        self.point = None

    def print(self):
        print("\nID = %s\n"
              "Fx = %s\n"
              "Fz = %s\n"
              "Ty = %s\n"
              "ux = %s\n"
              "uz = %s\n"
              "phiy = %s" % (self.ID, self.Fx, self.Fz, self.Ty, self.ux, self.uz, self.phiy))