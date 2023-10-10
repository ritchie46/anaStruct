from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np


class Vertex:
    """
    Utility point in 2D.
    """

    def __init__(
        self,
        x: Union[Vertex, Sequence[int], Sequence[float], np.ndarray, int, float],
        y: Union[int, float, None] = None,
    ):
        """Create a Vertex object

        Args:
            x (Union[Vertex, Sequence[int], Sequence[float], np.ndarray, int, float]): X coordinate
            y (Union[int, float, None], optional): Y coordinate. Defaults to None.
        """
        if isinstance(x, (Sequence)):
            self.coordinates: np.ndarray = np.array([x[0], x[1]], dtype=np.float32)
        elif isinstance(x, np.ndarray):
            self.coordinates = np.array(x, dtype=np.float32)
        elif isinstance(x, Vertex):
            self.coordinates = np.array(x.coordinates, dtype=np.float32)
        else:
            self.coordinates = np.array([x, y], dtype=np.float32)

    @property
    def x(self) -> float:
        """X coordinate

        Returns:
            float: X coordinate
        """
        return float(self.coordinates[0])

    @property
    def y(self) -> float:
        """Y coordinate

        Returns:
            float: Y coordinate
        """
        return float(self.coordinates[1])

    @property
    def z(self) -> float:
        """Z coordinate (equals negative of Y coordinate)

        Returns:
            float: Z coordinate
        """
        return float(self.coordinates[1] * -1)

    def modulus(self) -> float:
        """Magnitude of the vector from the origin to the Vertex

        Returns:
            float: Magnitude of the vector from the origin to the Vertex
        """
        return float(np.sqrt(np.sum(self.coordinates**2)))

    def unit(self) -> Vertex:
        """Unit vector from the origin to the Vertex

        Returns:
            Vertex: Unit vector from the origin to the Vertex
        """
        return (1 / self.modulus()) * self

    def displace_polar(
        self, alpha: float, radius: float, inverse_z_axis: bool = False
    ) -> None:
        """Displace the Vertex by a polar coordinate

        Args:
            alpha (float): Angle in radians
            radius (float): Radius
            inverse_z_axis (bool, optional): Return a Z coordinate (negative of Y coordinate)?. Defaults to False.
        """
        if inverse_z_axis:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] -= math.sin(alpha) * radius
        else:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] += math.sin(alpha) * radius

    def __add__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Add two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to add

        Returns:
            Vertex: Sum of the two Vertex objects
        """
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates + other
        return Vertex(coordinates)

    def __radd__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Add two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to add

        Returns:
            Vertex: Sum of the two Vertex objects
        """
        return self.__add__(other)

    def __sub__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Subtract two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to subtract

        Returns:
            Vertex: Difference of the two Vertex objects
        """
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates - other
        return Vertex(coordinates)

    def __rsub__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Subtract two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to subtract

        Returns:
            Vertex: Difference of the two Vertex objects
        """
        return self.__sub__(other)

    def __mul__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Multiply two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to multiply

        Returns:
            Vertex: Product of the two Vertex objects
        """
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates * other
        return Vertex(coordinates)

    def __rmul__(self, other: Union[Vertex, tuple, list, np.ndarray, float]) -> Vertex:
        """Multiply two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to multiply

        Returns:
            Vertex: Product of the two Vertex objects
        """
        return self.__mul__(other)

    def __truediv__(
        self, other: Union[Vertex, tuple, list, np.ndarray, float]
    ) -> Vertex:
        """Divide two Vertex objects

        Args:
            other (Union[Vertex, tuple, list, np.ndarray, float]): Vertex to divide

        Returns:
            Vertex: Quotient of the two Vertex objects
        """
        if isinstance(other, (tuple, list)):
            other = np.asarray(other)
        if isinstance(other, Vertex):
            other = other.coordinates

        coordinates = self.coordinates / other
        return Vertex(coordinates)

    def __eq__(self, other: object) -> bool | NotImplementedError:
        """Check if two Vertex objects are equal

        Args:
            other (object): Object to compare

        Returns:
            bool: True if the two Vertex objects are equal
        """
        if isinstance(other, Vertex):
            return self.x == other.x and self.y == other.y
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        return NotImplemented

    def __str__(self) -> str:
        """String representation of the Vertex object

        Returns:
            str: String representation of the Vertex object
        """
        return f"Vertex({self.x}, {self.y})"

    def __hash__(self) -> int:
        """Hash of the Vertex object

        Returns:
            int: Hash of the Vertex object
        """
        return hash(str(self))


def vertex_range(v1: Vertex, v2: Vertex, n: int) -> list:
    """Create a list of n Vertex objects between two Vertex objects

    Args:
        v1 (Vertex): Starting Vertex
        v2 (Vertex): Ending Vertex
        n (int): Number of Vertex objects to create

    Returns:
        list: List of n Vertex objects between v1 and v2
    """
    dv = v2 - v1
    return [v1 + dv * i / n for i in range(n + 1)]
