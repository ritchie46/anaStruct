class Node:
    def __init__(self, ID, Fx=0, Fz=0, Ty=0, ux=0, uz=0, phiy=0):
        self.ID = ID
        # forces
        self.Fx = Fx
        self.Fz = Fz
        self.Ty = Ty
        # displacements
        self.ux = ux
        self.uz = uz
        self.phiy = phiy

    def __sub__(self, other):
        Fx = self.Fx - other.Fx
        Fz = self.Fz - other.Fz
        Ty = self.Ty - other.Ty
        ux = self.ux - other.ux
        uz = self.uz - other.uz
        phiy = self.phiy - other.phiy

        return Nodes(self.ID, Fx, Fz, Ty, ux, uz, phiy)

    def print(self):
        print("\nID = %s\n"
              "Fx = %s\n"
              "Fz = %s\n"
              "Ty = %s\n"
              "ux = %s\n"
              "uz = %s\n"
              "phiy = %s" % (self.ID, self.Fx, self.Fz, self.Ty, self.ux, self.uz, self.phiy))