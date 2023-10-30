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
        Fy: float = 0.0,
        Tz: float = 0.0,
        ux: float = 0.0,
        uy: float = 0.0,
        phi_z: float = 0.0,
        vertex: Vertex = Vertex(0, 0),
        hinge: bool = False,
    ):
        """Create a Node object

        Args:
            id (int): ID of the node
            Fx (float, optional): Value of Fx force. Defaults to 0.0.
            Fy (float, optional): Value of Fy force. Defaults to 0.0.
            Tz (float, optional): Value of Tz moment. Defaults to 0.0.
            ux (float, optional): Value of ux displacement. Defaults to 0.0.
            uy (float, optional): Value of uy displacement. Defaults to 0.0.
            phi_z (float, optional): Value of phi_z rotation. Defaults to 0.0.
            vertex (Vertex, optional): Point object coordinate. Defaults to Vertex(0, 0).
            hinge (bool, optional): Is this node a hinge. Defaults to False.
        """
        self.id = id
        # forces
        self.Fx = Fx
        self.Fy = Fy
        self.Tz = Tz
        # displacements
        self.ux = ux
        self.uy = uy
        self.phi_z = phi_z
        self.vertex = vertex
        self.hinge = hinge
        self.elements: Dict[int, Element] = {}

    @property
    def Fy_neg(self) -> float:
        """Fy is the vertical force, and the negative of Fy

        Returns:
            float: negative of Fy
        """
        return -self.Fy

    def __str__(self) -> str:
        """String representation of the node

        Returns:
            str: String representation of the node
        """
        if self.vertex:
            return (
                f"[id = {self.id}, Fx = {self.Fx}, Fy = {self.Fy}, Tz = {self.Tz}, ux = {self.ux}, "
                f"uy = {self.uy}, phi_z = {self.phi_z}, x = {self.vertex.x}, y = {self.vertex.y}]"
            )
        return (
            f"[id = {self.id}, Fx = {self.Fx}, Fy = {self.Fy}, Tz = {self.Tz}, ux = {self.ux}, "
            f"uy = {self.uy}, phi_z = {self.phi_z}]"
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
        Fy = self.Fy + other.Fy
        Tz = self.Tz + other.Tz

        return Node(
            id=self.id,
            Fx=Fx,
            Fy=Fy,
            Tz=Tz,
            ux=self.ux,
            uy=self.uy,
            phi_z=self.phi_z,
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
        Fy = self.Fy - other.Fy
        Tz = self.Tz - other.Tz

        return Node(
            self.id,
            Fx,
            Fy,
            Tz,
            self.ux,
            self.uy,
            self.phi_z,
            self.vertex,
            hinge=self.hinge,
        )

    def reset(self) -> None:
        """Reset the node to zero forces and displacements"""
        self.Fx = self.Fy = self.Tz = self.ux = self.uy = self.phi_z = 0
        self.hinge = False

    def add_results(self, other: Node) -> None:
        """Add the results of another node to this node

        Args:
            other (Node): Node to add
        """
        assert (
            self.id == other.id
        ), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        assert self.phi_z is not None
        assert self.ux is not None
        assert self.uy is not None
        assert other.phi_z is not None
        assert other.ux is not None
        assert other.uy is not None
        self.Fx = self.Fx + other.Fx
        self.Fy = self.Fy + other.Fy
        self.Tz = self.Tz + other.Tz
        self.ux = self.ux + other.ux
        self.uy = self.uy + other.uy
        self.phi_z = self.phi_z + other.phi_z
