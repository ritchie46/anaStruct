from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from anastruct.vertex import Vertex

if TYPE_CHECKING:
    from anastruct.fem.elements import Element


class Node:
    def __init__(
        self,
        id: int,  # pylint: disable=redefined-builtin
        Fx: float = 0.0,
        Fz: float = 0.0,
        Ty: float = 0.0,
        ux: float = 0.0,
        uz: float = 0.0,
        phi_y: float = 0,
        vertex: Vertex = Vertex(0, 0),
        hinge: bool = False,
    ):
        """
        :param id: ID of the node, integer
        :param Fx: Value of Fx
        :param Fz: Value of Fz
        :param Ty: Value of Ty
        :param ux: Value of ux
        :param uz: Value of uz
        :param phi_y: Value of phi
        :param vertex: Point object
        :param hinge: Boolean
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
        self.hinge = hinge
        self.elements: Dict[int, Element] = {}

    @property
    def Fy(self) -> float:
        return -self.Fz

    def __str__(self):
        if self.vertex:
            return (
                f"[id = {self.id}, Fx = {self.Fx}, Fz = {self.Fz}, Ty = {self.Ty}, ux = {self.ux}, "
                f"uz = {self.uz}, phi_y = {self.phi_y}, x = {self.vertex.x}, y = {self.vertex.y}]"
            )
        else:
            return (
                f"[id = {self.id}, Fx = {self.Fx}, Fz = {self.Fz}, Ty = {self.Ty}, ux = {self.ux}, "
                f"uz = {self.uz}, phi_y = {self.phi_y}]"
            )

    def __add__(self, other: Node) -> Node:
        assert (
            self.id == other.id
        ), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx + other.Fx
        Fz = self.Fz + other.Fz
        Ty = self.Ty + other.Ty

        return Node(
            id=self.id,
            Fx=Fx,
            Fz=Fz,
            Ty=Ty,
            ux=self.ux,
            uz=self.uz,
            phi_y=self.phi_y,
            vertex=self.vertex,
            hinge=self.hinge,
        )

    def __sub__(self, other: Node) -> Node:
        assert (
            self.id == other.id
        ), "Cannot subtract nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx - other.Fx
        Fz = self.Fz - other.Fz
        Ty = self.Ty - other.Ty

        return Node(
            self.id,
            Fx,
            Fz,
            Ty,
            self.ux,
            self.uz,
            self.phi_y,
            self.vertex,
            hinge=self.hinge,
        )

    def reset(self):
        self.Fx = self.Fz = self.Ty = self.ux = self.uz = self.phi_y = 0
        self.hinge = False
