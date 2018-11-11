import pprint


class LoadCase:
    def __init__(self, name):
        self.name = name
        self.spec = dict()

    def q_load(self, q, element_id, direction="element"):
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"
        """
        self.spec['q_load'] = dict(q=q, element_id=element_id, direction=direction)

    def point_load(self, node_id, Fx=0, Fz=0, rotation=0):
        """
        Apply a point load to a node.

        :param node_id: (int/ list) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fz: (flt/ list) Force in global x direction.
        :param rotation: (flt/ list) Rotate the force clockwise. Rotation is in degrees.
        """
        self.spec['point_load'] = dict(node_id=node_id, Fx=Fx, Fz=Fz, rotation=rotation)

    def moment_load(self, node_id, Ty):
        """
        Apply a moment on a node.

        :param node_id: (int/ list) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        self.spec['moment_load'] = dict(node_id=node_id, Ty=Ty)

    def dead_load(self, element_id, g):
        """
        Apply a dead load in kN/m on elements.

        :param element_id: (int/ list) representing the element ID
        :param g: (flt/ list) Weight per meter. [kN/m] / [N/m]
        """
        self.spec['dead_load'] = dict(element_id=element_id, g=g)

    def __str__(self):
        return f'Loadcase {self.name}:\n' + pprint.pformat(self.spec)

