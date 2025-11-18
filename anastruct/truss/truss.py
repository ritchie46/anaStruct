from typing import Literal, Optional

import numpy as np

from anastruct.truss.truss_class import FlatTruss, RoofTruss
from anastruct.types import SectionProps, Vertex


class HoweFlatTruss(FlatTruss):
    @property
    def type(self) -> str:
        return "Howe Flat Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(0.0, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(0, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(self.width, self.height))

    def define_connectivity(self) -> None:
        n_bottom_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_up" else 0)
        )
        n_top_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_down" else 0)
        )

        # Bottom chord connectivity
        self.bottom_chord_node_ids = [list(range(0, n_bottom_nodes))]

        # Top chord connectivity
        self.top_chord_node_ids = [
            list(range(n_bottom_nodes, n_bottom_nodes + n_top_nodes))
        ]

        # Web diagonals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None  # a None index means go to the end
        end_top = None
        if self.end_type == "triangle_up":
            # special case: end diagonal slopes in the opposite direction
            self.web_node_pairs.append((0, n_bottom_nodes))
            self.web_node_pairs.append(
                (n_bottom_nodes - 1, n_bottom_nodes + n_top_nodes - 1)
            )
            start_top = 2
            end_top = -3
        elif self.end_type == "flat":
            start_top = 1
            end_top = -2
        mid_bot = len(self.bottom_chord_node_ids) // 2
        mid_top = len(self.top_chord_node_ids) // 2
        for b, t in zip(
            self.bottom_chord_node_ids[0][start_bot : mid_bot + 1],
            self.top_chord_node_ids[0][start_top : mid_top + 1],
        ):
            self.web_node_pairs.append((b, t))
        for b, t in zip(
            self.bottom_chord_node_ids[0][end_bot : mid_bot - 1 : -1],
            self.top_chord_node_ids[0][end_top : mid_top - 1 : -1],
        ):
            self.web_node_pairs.append((b, t))

        # Web verticals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None
        end_top = None
        if self.end_type == "triangle_up":
            start_top = 1
            end_top = -1
        elif self.end_type == "triangle_down":
            start_bot = 1
            end_bot = -1
        for b, t in zip(
            self.bottom_chord_node_ids[0][start_bot:end_bot],
            self.top_chord_node_ids[0][start_top:end_top],
        ):
            self.web_verticals_node_pairs.append((b, t))


class PrattFlatTruss(FlatTruss):
    @property
    def type(self) -> str:
        return "Pratt Flat Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(0.0, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(0, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(self.width, self.height))

    def define_connectivity(self) -> None:
        n_bottom_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_up" else 0)
        )
        n_top_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_down" else 0)
        )

        # Bottom chord connectivity
        self.bottom_chord_node_ids = [list(range(0, n_bottom_nodes))]

        # Top chord connectivity
        self.top_chord_node_ids = [
            list(range(n_bottom_nodes, n_bottom_nodes + n_top_nodes))
        ]

        # Web diagonals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None  # a None index means go to the end
        end_top = None
        if self.end_type == "triangle_down":
            # special case: end diagonal slopes in the opposite direction
            self.web_node_pairs.append((n_bottom_nodes, 0))
            self.web_node_pairs.append(
                (n_bottom_nodes + n_top_nodes - 1, n_bottom_nodes - 1)
            )
            start_bot = 2
            end_bot = -3
        elif self.end_type == "flat":
            start_bot = 1
            end_bot = -2
        mid_bot = len(self.bottom_chord_node_ids) // 2
        mid_top = len(self.top_chord_node_ids) // 2
        for b, t in zip(
            self.bottom_chord_node_ids[0][start_bot : mid_bot + 1],
            self.top_chord_node_ids[0][start_top : mid_top + 1],
        ):
            self.web_node_pairs.append((b, t))
        for b, t in zip(
            self.bottom_chord_node_ids[0][end_bot : mid_bot - 1 : -1],
            self.top_chord_node_ids[0][end_top : mid_top - 1 : -1],
        ):
            self.web_node_pairs.append((b, t))

        # Web verticals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None
        end_top = None
        if self.end_type == "triangle_up":
            start_top = 1
            end_top = -1
        elif self.end_type == "triangle_down":
            start_bot = 1
            end_bot = -1
        for b, t in zip(
            self.bottom_chord_node_ids[0][start_bot:end_bot],
            self.top_chord_node_ids[0][start_top:end_top],
        ):
            self.web_verticals_node_pairs.append((b, t))


class WarrenFlatTruss(FlatTruss):
    # Data types specific to this truss type
    EndType = Literal["triangle_down", "triangle_up"]
    SupportLoc = Literal["bottom_chord", "top_chord", "both"]

    # Additional geometry for this truss type
    unit_width: float
    end_type: EndType
    supports_loc: SupportLoc

    # Computed properties
    n_units: int
    end_width: float

    @property
    def type(self) -> str:
        return "Warren Flat Truss"

    def __init__(
        self,
        width: float,
        height: float,
        unit_width: float,
        end_type: EndType = "triangle_down",
        supports_loc: SupportLoc = "bottom_chord",
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        # Note that the maths for a Warren truss is simpler than for Howe/Pratt, because there
        # cannot be any option for non-even number of units, and there are no special cases for
        # web verticals.
        min_end_fraction = 0.5  # Not used for Warren truss
        enforce_even_units = True  # Handled internally for Warren truss
        super().__init__(
            width,
            height,
            unit_width,
            end_type,
            supports_loc,
            min_end_fraction,
            enforce_even_units,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )
        self.end_width = (width - self.n_units * unit_width) / 2 + (unit_width / 2)

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type == "triangle_down":
            self.nodes.append(Vertex(0.0, 0.0))
        else:
            self.nodes.append(Vertex(self.end_width - self.unit_width / 2, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type == "triangle_down":
            self.nodes.append(Vertex(self.width, 0.0))
        else:
            self.nodes.append(
                Vertex(self.width - (self.end_width - self.unit_width / 2), 0.0)
            )

        # Top chord nodes
        if self.end_type == "triangle_up":
            self.nodes.append(Vertex(0, self.height))
        else:
            self.nodes.append(Vertex(self.end_width - self.unit_width / 2, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type == "triangle_up":
            self.nodes.append(Vertex(self.width, self.height))
        else:
            self.nodes.append(
                Vertex(self.width - (self.end_width - self.unit_width / 2), self.height)
            )

    def define_connectivity(self) -> None:
        n_bottom_nodes = int(self.n_units) + (
            1 if self.end_type == "triangle_down" else 0
        )
        n_top_nodes = int(self.n_units) + (1 if self.end_type == "triangle_up" else 0)

        # Bottom chord connectivity
        self.bottom_chord_node_ids = [list(range(0, n_bottom_nodes))]

        # Top chord connectivity
        self.top_chord_node_ids = [
            list(range(n_bottom_nodes, n_bottom_nodes + n_top_nodes))
        ]

        # Web diagonals connectivity
        # sloping up from bottom left to top right
        top_start = 0 if self.end_type == "triangle_down" else 1
        for b, t in zip(
            self.bottom_chord_node_ids[0],
            self.top_chord_node_ids[0][top_start:],
        ):
            self.web_node_pairs.append((b, t))
        # sloping down from top left to bottom right
        bot_start = 0 if self.end_type == "triangle_up" else 1
        for b, t in zip(
            self.top_chord_node_ids[0],
            self.bottom_chord_node_ids[0][bot_start:],
        ):
            self.web_node_pairs.append((b, t))


class KingPostRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "King Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, self.height))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2]]
        left_v = 0
        right_v = 2

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 3], [3, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 4)  # left overhang
            self.top_chord_node_ids[1].append(5)  # right overhang

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 3))  # center vertical


class QueenPostRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Queen Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2]]
        left_v = 0
        right_v = 2

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 3, 4], [4, 5, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 6)  # left overhang
            self.top_chord_node_ids[1].append(7)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 3))  # left diagonal
        self.web_node_pairs.append((1, 5))  # right diagonal

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 3))  # center vertical


class FinkRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Fink Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3]]
        left_v = 0
        right_v = 3

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 4, 5], [5, 6, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 7)  # left overhang
            self.top_chord_node_ids[1].append(8)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 4))
        self.web_node_pairs.append((1, 5))
        self.web_node_pairs.append((2, 5))
        self.web_node_pairs.append((2, 6))


class HoweRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Howe Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 5, 6], [6, 7, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 8)  # left overhang
            self.top_chord_node_ids[1].append(9)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 5))  # left diagonal
        self.web_node_pairs.append((2, 7))  # right diagonal

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 5))  # left vertical
        self.web_verticals_node_pairs.append((2, 6))  # centre vertical
        self.web_verticals_node_pairs.append((3, 7))  # right vertical


class PrattRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Pratt Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 5, 6], [6, 7, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 8)  # left overhang
            self.top_chord_node_ids[1].append(9)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 6))  # left diagonal
        self.web_node_pairs.append((3, 6))  # right diagonal

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 5))  # left vertical
        self.web_verticals_node_pairs.append((2, 6))  # centre vertical
        self.web_verticals_node_pairs.append((3, 7))  # right vertical


class FanRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Fan Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3]]
        left_v = 0
        right_v = 3

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 4, 5, 6], [6, 7, 8, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 9)  # left overhang
            self.top_chord_node_ids[1].append(10)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 4))
        self.web_node_pairs.append((1, 6))
        self.web_node_pairs.append((2, 6))
        self.web_node_pairs.append((2, 8))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 5))
        self.web_verticals_node_pairs.append((2, 7))


class ModifiedQueenPostRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Modified Queen Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 5, 6, 7], [7, 8, 9, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 10)  # left overhang
            self.top_chord_node_ids[1].append(11)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 5))
        self.web_node_pairs.append((1, 6))
        self.web_node_pairs.append((2, 6))
        self.web_node_pairs.append((2, 8))
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((3, 9))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 7))  # center vertical


class DoubleFinkRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Double Fink Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 5, 0.0))
        self.nodes.append(Vertex(2 * self.width / 5, 0.0))
        self.nodes.append(Vertex(3 * self.width / 5, 0.0))
        self.nodes.append(Vertex(4 * self.width / 5, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4, 5]]
        left_v = 0
        right_v = 5

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 6, 7, 8], [8, 9, 10, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 11)  # left overhang
            self.top_chord_node_ids[1].append(12)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 6))
        self.web_node_pairs.append((1, 7))
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((2, 8))
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((3, 9))
        self.web_node_pairs.append((4, 9))
        self.web_node_pairs.append((4, 10))


class DoubleHoweRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Double Howe Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, 0.0))
        self.nodes.append(Vertex(2 * self.width / 6, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(4 * self.width / 6, 0.0))
        self.nodes.append(Vertex(5 * self.width / 6, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4, 5, 6]]
        left_v = 0
        right_v = 6

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 7, 8, 9], [9, 10, 11, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 12)  # left overhang
            self.top_chord_node_ids[1].append(13)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((3, 10))
        self.web_node_pairs.append((4, 11))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 7))
        self.web_verticals_node_pairs.append((2, 8))
        self.web_verticals_node_pairs.append((3, 9))  # center vertical
        self.web_verticals_node_pairs.append((4, 10))
        self.web_verticals_node_pairs.append((5, 11))


class ModifiedFanRoofTruss(RoofTruss):
    @property
    def type(self) -> str:
        return "Modified Fan Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 8, 1 * self.height / 4))
        self.nodes.append(Vertex(2 * self.width / 8, 2 * self.height / 4))
        self.nodes.append(Vertex(3 * self.width / 8, 3 * self.height / 4))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(5 * self.width / 8, 3 * self.height / 4))
        self.nodes.append(Vertex(6 * self.width / 8, 2 * self.height / 4))
        self.nodes.append(Vertex(7 * self.width / 8, 1 * self.height / 4))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = [[left_v, 5, 6, 7, 8], [8, 9, 10, 11, right_v]]
        if self.overhang_length > 0:
            self.top_chord_node_ids[0].insert(0, 12)  # left overhang
            self.top_chord_node_ids[1].append(13)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((1, 5))
        self.web_node_pairs.append((1, 7))
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((2, 9))
        self.web_node_pairs.append((3, 9))
        self.web_node_pairs.append((3, 11))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((1, 6))
        self.web_verticals_node_pairs.append((2, 8))  # center vertical
        self.web_verticals_node_pairs.append((3, 10))


class AtticRoofTruss(RoofTruss):
    # Additional properties for this truss type
    attic_width: float
    attic_height: float

    # Computed properties for this truss type
    wall_x: float
    wall_y: float
    ceiling_y: float
    ceiling_x: float
    wall_ceiling_intersect: bool = False

    @property
    def type(self) -> str:
        return "Attic Roof Truss"

    def __init__(
        self,
        width: float,
        roof_pitch_deg: float,
        attic_width: float,
        attic_height: Optional[float] = None,
        overhang_length: float = 0.0,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        super().__init__(
            width=width,
            roof_pitch_deg=roof_pitch_deg,
            overhang_length=overhang_length,
            top_chord_section=top_chord_section,
            bottom_chord_section=bottom_chord_section,
            web_section=web_section,
            web_verticals_section=web_verticals_section,
        )
        self.attic_width = attic_width
        self.wall_x = self.width / 2 - self.attic_width / 2
        self.wall_y = self.wall_x * np.tan(self.roof_pitch)
        if attic_height is None:
            self.attic_height = self.wall_y
        else:
            self.attic_height = attic_height
        self.ceiling_y = self.attic_height
        self.ceiling_x = self.width / 2 - (self.height - self.ceiling_y) / np.tan(
            self.roof_pitch
        )
        if self.ceiling_y == self.wall_y:
            self.wall_ceiling_intersect = True
        if self.ceiling_y < self.wall_y or self.ceiling_x < self.wall_y:
            raise ValueError(
                "Attic height may not be less than the attic wall height. Please increase your attic width and/or attic height."
            )

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.wall_x, 0.0))
        self.nodes.append(Vertex(self.width - self.wall_x, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.wall_x / 2, self.wall_y / 2))
        self.nodes.append(Vertex(self.wall_x, self.wall_y))
        if not self.wall_ceiling_intersect:
            self.nodes.append(Vertex(self.ceiling_x, self.ceiling_y))
        self.nodes.append(Vertex(self.width / 2, self.height))
        if not self.wall_ceiling_intersect:
            self.nodes.append(Vertex(self.width - self.ceiling_x, self.ceiling_y))
        self.nodes.append(Vertex(self.width - self.wall_x, self.wall_y))
        self.nodes.append(Vertex(self.width - self.wall_x / 2, self.wall_y / 2))
        self.nodes.append(
            Vertex(self.width / 2, self.ceiling_y)
        )  # special node in the middle of the ceiling beam
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [[0, 1, 2, 3]]
        left_v = 0
        right_v = 3

        if self.wall_ceiling_intersect:
            # Top chord connectivity (left and right slopes stored separately)
            self.top_chord_node_ids = [
                [left_v, 4, 5, 6],
                [6, 7, 8, right_v],
                [5, 9, 7],  # attic ceiling
            ]
            if self.overhang_length > 0:
                self.top_chord_node_ids[0].insert(0, 10)  # left overhang
                self.top_chord_node_ids[1].append(11)  # right overhang

            # Web diagonals connectivity
            self.web_node_pairs.append((1, 4))
            self.web_node_pairs.append(
                (9, 6)
            )  # special case: this is actually the center vertical post
            self.web_node_pairs.append((2, 8))

            # Web verticals connectivity
            self.web_verticals_node_pairs.append((1, 5))
            self.web_verticals_node_pairs.append((2, 7))

        else:
            # Top chord connectivity (left and right slopes stored separately)
            self.top_chord_node_ids = [
                [left_v, 4, 5, 6, 7],
                [7, 8, 9, 10, right_v],
                [6, 11, 8],  # attic ceiling
            ]
            if self.overhang_length > 0:
                self.top_chord_node_ids[0].insert(0, 12)  # left overhang
                self.top_chord_node_ids[1].append(13)  # right overhang

            # Web diagonals connectivity
            self.web_node_pairs.append((1, 4))
            self.web_node_pairs.append(
                (11, 7)
            )  # special case: this is actually the center vertical post
            self.web_node_pairs.append((2, 10))

            # Web verticals connectivity
            self.web_verticals_node_pairs.append((1, 5))
            self.web_verticals_node_pairs.append((2, 9))


__all__ = [
    "HoweFlatTruss",
    "PrattFlatTruss",
    "WarrenFlatTruss",
    "KingPostRoofTruss",
    "QueenPostRoofTruss",
    "FinkRoofTruss",
    "HoweRoofTruss",
    "PrattRoofTruss",
    "FanRoofTruss",
    "ModifiedQueenPostRoofTruss",
    "DoubleFinkRoofTruss",
    "DoubleHoweRoofTruss",
    "ModifiedFanRoofTruss",
    "AtticRoofTruss",
]
