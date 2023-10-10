from __future__ import annotations

from typing import TYPE_CHECKING, Dict

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
        phi_y: float = 0.0,
        vertex: Vertex = Vertex(0, 0),
        hinge: bool = False,
    ):
        """Create a Node object

        Args:
            id (int): ID of the node
            Fx (float, optional): Value of Fx force. Defaults to 0.0.
            Fz (float, optional): Value of Fz force. Defaults to 0.0.
            Ty (float, optional): Value of Ty moment. Defaults to 0.0.
            ux (float, optional): Value of ux displacement. Defaults to 0.0.
            uz (float, optional): Value of uz displacement. Defaults to 0.0.
            phi_y (float, optional): Value of phi_y rotation. Defaults to 0.0.
            vertex (Vertex, optional): Point object coordinate. Defaults to Vertex(0, 0).
            hinge (bool, optional): Is this node a hinge. Defaults to False.
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
        """Fy is the vertical force, and the negative of Fz

        Returns:
            float: negative of Fz
        """
        return -self.Fz

    def __str__(self) -> str:
        """String representation of the node

        Returns:
            str: String representation of the node
        """
        if self.vertex:
            return (
                f"[id = {self.id}, Fx = {self.Fx}, Fz = {self.Fz}, Ty = {self.Ty}, ux = {self.ux}, "
                f"uz = {self.uz}, phi_y = {self.phi_y}, x = {self.vertex.x}, y = {self.vertex.y}]"
            )
        return (
            f"[id = {self.id}, Fx = {self.Fx}, Fz = {self.Fz}, Ty = {self.Ty}, ux = {self.ux}, "
            f"uz = {self.uz}, phi_y = {self.phi_y}]"
        )

    def __add__(self, other: Node) -> Node:
        """Add two nodes

        Args:
            other (Node): Node to add

        Returns:
            Node: Sum of the two nodes
        """
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
        """Subtract two nodes

        Args:
            other (Node): Node to subtract

        Returns:
            Node: Difference of the two nodes
        """
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

    def reset(self) -> None:
        """Reset the node to zero forces and displacements"""
        self.Fx = self.Fz = self.Ty = self.ux = self.uz = self.phi_y = 0
        self.hinge = False

    def add_results(self, other: Node) -> None:
        """Add the results of another node to this node

        Args:
            other (Node): Node to add
        """
        assert (
            self.id == other.id
        ), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        assert self.phi_y is not None
        assert self.ux is not None
        assert self.uz is not None
        assert other.phi_y is not None
        assert other.ux is not None
        assert other.uz is not None
        self.Fx = self.Fx + other.Fx
        self.Fz = self.Fz + other.Fz
        self.Ty = self.Ty + other.Ty
        self.ux = self.ux + other.ux
        self.uz = self.uz + other.uz
        self.phi_y = self.phi_y + other.phi_y
