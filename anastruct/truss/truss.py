from typing import Literal, Optional

import numpy as np

from anastruct.truss.truss_class import DEFAULT_TRUSS_SECTION, RoofTruss, Truss
from anastruct.types import SectionProps, Vertex


class HoweFlatTruss(Truss):
    # Data types specific to this truss type
    EndType = Literal["flat", "triangle_down", "triangle_up"]
    SupportLoc = Literal["bottom_chord", "top_chord", "both"]

    # Additional geometry for this truss type
    unit_width: float
    end_type: EndType
    supports_loc: SupportLoc

    # Addtional configuration
    min_end_fraction: float
    enforce_even_units: bool

    # Computed properties
    n_units: int
    end_width: float

    @property
    def type(self) -> str:
        return "Howe Flat Truss"

    def __init__(
        self,
        width: float,
        height: float,
        unit_width: float,
        end_type: EndType = "triangle_down",
        supports_loc: SupportLoc = "bottom_chord",
        min_end_fraction: float = 0.5,
        enforce_even_units: bool = True,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):

        self.unit_width = unit_width
        self.end_type = end_type
        self.supports_loc = supports_loc
        self.min_end_fraction = min_end_fraction
        self.enforce_even_units = enforce_even_units
        self.n_units = np.floor(
            (width - unit_width * 2 * min_end_fraction) / unit_width
        )
        if self.enforce_even_units and self.n_units % 2 != 0:
            self.n_units -= 1
        self.end_width = (width - self.n_units * unit_width) / 2
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

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

    def define_supports(self) -> None:
        bottom_left = 0
        bottom_right = max(self.bottom_chord_node_ids[0])
        top_left = min(self.top_chord_node_ids[0])
        top_right = max(self.top_chord_node_ids[0])
        if self.supports_loc in ["bottom_chord", "both"]:
            self.support_definitions[bottom_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[bottom_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )
        if self.supports_loc in ["top_chord", "both"]:
            self.support_definitions[top_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[top_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )


class PrattFlatTruss(Truss):
    # Data types specific to this truss type
    EndType = Literal["flat", "triangle_down", "triangle_up"]
    SupportLoc = Literal["bottom_chord", "top_chord", "both"]

    # Additional geometry for this truss type
    unit_width: float
    end_type: EndType
    supports_loc: SupportLoc

    # Addtional configuration
    min_end_fraction: float
    enforce_even_units: bool

    # Computed properties
    n_units: int
    end_width: float

    @property
    def type(self) -> str:
        return "Pratt Flat Truss"

    def __init__(
        self,
        width: float,
        height: float,
        unit_width: float,
        end_type: EndType = "triangle_down",
        supports_loc: SupportLoc = "bottom_chord",
        min_end_fraction: float = 0.5,
        enforce_even_units: bool = True,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        self.unit_width = unit_width
        self.end_type = end_type
        self.supports_loc = supports_loc
        self.min_end_fraction = min_end_fraction
        self.enforce_even_units = enforce_even_units
        self.n_units = np.floor(
            (width - unit_width * 2 * min_end_fraction) / unit_width
        )
        if self.enforce_even_units and self.n_units % 2 != 0:
            self.n_units -= 1
        self.end_width = (width - self.n_units * unit_width) / 2
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

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

    def define_supports(self) -> None:
        bottom_left = 0
        bottom_right = max(self.bottom_chord_node_ids[0])
        top_left = min(self.top_chord_node_ids[0])
        top_right = max(self.top_chord_node_ids[0])
        if self.supports_loc in ["bottom_chord", "both"]:
            self.support_definitions[bottom_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[bottom_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )
        if self.supports_loc in ["top_chord", "both"]:
            self.support_definitions[top_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[top_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )


class WarrenFlatTruss(Truss):
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
        self.unit_width = unit_width
        self.end_type = end_type
        self.supports_loc = supports_loc
        # Note that the maths for a Warren truss is simpler than for Howe/Pratt, because there
        # cannot be any option for non-even number of units, and there are no special cases for
        # web verticals.
        self.n_units = np.floor(width / unit_width)
        if self.n_units % 2 != 0:
            self.n_units -= 1
        self.end_width = (width - self.n_units * unit_width) / 2 + (unit_width / 2)
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

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

    def define_supports(self) -> None:
        bottom_left = 0
        bottom_right = max(self.bottom_chord_node_ids[0])
        top_left = min(self.top_chord_node_ids[0])
        top_right = max(self.top_chord_node_ids[0])
        if self.supports_loc in ["bottom_chord", "both"]:
            self.support_definitions[bottom_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[bottom_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )
        if self.supports_loc in ["top_chord", "both"]:
            self.support_definitions[top_left] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[top_right] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )


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
        self.nodes.append(Vertex(self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
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
        self.nodes.append(Vertex(self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
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
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 2

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
        self.nodes.append(Vertex(self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
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
        self.bottom_chord_node_ids = [[0, 1, 2, 3, 4]]
        left_v = 0
        right_v = 2

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
        self.nodes.append(Vertex(self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 6, self.height / 3))
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
]
