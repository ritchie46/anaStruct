from typing import Literal, Optional

import numpy as np

from anastruct.truss.truss_class import DEFAULT_TRUSS_SECTION, Truss
from anastruct.types import SectionProps, Vertex


class HoweTruss(Truss):
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
        return "Howe"

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
        self.bottom_chord_node_ids = list(range(0, n_bottom_nodes))

        # Top chord connectivity
        self.top_chord_node_ids = list(
            range(n_bottom_nodes, n_bottom_nodes + n_top_nodes)
        )

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
            end_top = -2
        elif self.end_type == "flat":
            start_top = 1
            end_top = -1
        mid_bot = len(self.bottom_chord_node_ids) // 2
        mid_top = len(self.top_chord_node_ids) // 2
        for b, t in zip(
            self.bottom_chord_node_ids[start_bot : mid_bot + 1],
            self.top_chord_node_ids[start_top : mid_top + 1],
        ):
            self.web_node_pairs.append((b, t))
        for b, t in zip(
            self.bottom_chord_node_ids[end_bot : mid_bot - 1 : -1],
            self.top_chord_node_ids[end_top : mid_top - 1 : -1],
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
            self.bottom_chord_node_ids[start_bot:end_bot],
            self.top_chord_node_ids[start_top:end_top],
        ):
            self.web_verticals_node_pairs.append((b, t))

    def define_supports(self) -> None:
        if self.supports_loc in ["bottom_chord", "both"]:
            self.support_definitions[0] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[max(self.bottom_chord_node_ids)] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )
        if self.supports_loc in ["top_chord", "both"]:
            self.support_definitions[min(self.top_chord_node_ids)] = (
                self.supports_type if self.supports_type != "simple" else "pinned"
            )
            self.support_definitions[max(self.top_chord_node_ids)] = (
                self.supports_type if self.supports_type != "simple" else "roller"
            )
