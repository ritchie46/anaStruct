from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np

from anastruct import SystemElements
from anastruct.fem.system_components.util import add_node
from anastruct.types import LoadDirection, SectionProps, Vertex

DEFAULT_TRUSS_SECTION: SectionProps = {
    "EI": 1e6,
    "EA": 1e8,
    "g": 0.0,
}


class Truss(ABC):
    # Common geometry
    width: float
    height: float

    # Material properties
    top_chord_section: SectionProps
    bottom_chord_section: SectionProps
    web_section: SectionProps
    web_verticals_section: SectionProps

    # Configuraion
    top_chord_continous: bool
    bottom_chord_continuous: bool
    supports_type: Literal["simple", "pinned", "fixed"]

    # Defined by subclass
    nodes: list[Vertex] = []
    top_chord_node_ids: list[list[int]] = []
    bottom_chord_node_ids: list[list[int]] = []
    web_node_pairs: list[tuple[int, int]] = []
    web_verticals_node_pairs: list[tuple[int, int]] = []
    support_definitions: dict[int, Literal["fixed", "pinned", "roller"]] = {}
    top_chord_length: float = 0.0
    bottom_chord_length: float = 0.0

    # System
    system: SystemElements

    def __init__(
        self,
        width: float,
        height: float,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
        top_chord_continous: bool = True,
        bottom_chord_continuous: bool = True,
        supports_type: Literal["simple", "pinned", "fixed"] = "simple",
    ):
        self.width = width
        self.height = height
        self.top_chord_section = top_chord_section or DEFAULT_TRUSS_SECTION
        self.bottom_chord_section = bottom_chord_section or DEFAULT_TRUSS_SECTION
        self.web_section = web_section or DEFAULT_TRUSS_SECTION
        self.web_verticals_section = web_verticals_section or self.web_section
        self.top_chord_continous = top_chord_continous
        self.bottom_chord_continuous = bottom_chord_continuous
        self.supports_type = supports_type

        self.define_nodes()
        self.define_connectivity()
        self.define_supports()

        self.system = SystemElements()
        self.add_nodes()
        self.add_elements()
        self.add_supports()

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def define_nodes(self) -> None:
        pass

    @abstractmethod
    def define_connectivity(self) -> None:
        pass

    @abstractmethod
    def define_supports(self) -> None:
        pass

    def add_nodes(self) -> None:
        for i, vertex in enumerate(self.nodes):
            add_node(self.system, point=vertex, node_id=i)

    def add_elements(self) -> None:
        def add_segment_elements(
            node_pairs: Iterable[tuple[int, int]],
            section: SectionProps,
            continuous: bool,
        ) -> None:
            for i, j in node_pairs:
                self.system.add_element(
                    location=(self.nodes[i], self.nodes[j]),
                    EA=section["EA"],
                    EI=section["EI"],
                    g=section["g"],
                    spring=None if continuous else {1: 0.0, 2: 0.0},
                )

        # Bottom chord elements
        for segment_node_ids in self.bottom_chord_node_ids:
            add_segment_elements(
                node_pairs=zip(segment_node_ids[:-1], segment_node_ids[1:]),
                section=self.bottom_chord_section,
                continuous=self.bottom_chord_continuous,
            )

        # Top chord elements
        for segment_node_ids in self.top_chord_node_ids:
            add_segment_elements(
                node_pairs=zip(segment_node_ids[:-1], segment_node_ids[1:]),
                section=self.top_chord_section,
                continuous=self.top_chord_continous,
            )

        # Web diagonal elements
        add_segment_elements(
            node_pairs=self.web_node_pairs,
            section=self.web_section,
            continuous=False,
        )

        # Web vertical elements
        add_segment_elements(
            node_pairs=self.web_verticals_node_pairs,
            section=self.web_verticals_section,
            continuous=False,
        )

    def add_supports(self) -> None:
        for node_id, support_type in self.support_definitions.items():
            if support_type == "fixed":
                self.system.add_support_fixed(node_id=node_id)
            elif support_type == "pinned":
                self.system.add_support_hinged(node_id=node_id)
            elif support_type == "roller":
                self.system.add_support_roll(node_id=node_id)

    def apply_q_loads_to_top_chord(
        self,
        q: Union[float, Sequence[float]],
        direction: Union["LoadDirection", Sequence["LoadDirection"]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        pass

    def apply_q_loads_to_bottom_chord(
        self,
        x_start: float,
        x_end: float,
        q: Union[float, Sequence[float]],
        direction: Union["LoadDirection", Sequence["LoadDirection"]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        pass

    def apply_point_load_to_top_chord(
        self, x_loc: float, Fx: float = 0.0, Fy: float = 0.0, rotation: float = 0.0
    ) -> None:
        pass

    def apply_point_load_to_bottom_chord(
        self, x_loc: float, Fx: float = 0.0, Fy: float = 0.0, rotation: float = 0.0
    ) -> None:
        pass

    def show_structure(self) -> None:
        self.system.show_structure()


class FlatTruss(Truss):
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
    @abstractmethod
    def type(self) -> str:
        return "[Generic] Flat Truss"

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

    @abstractmethod
    def define_nodes(self) -> None:
        pass

    @abstractmethod
    def define_connectivity(self) -> None:
        pass

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


class RoofTruss(Truss):
    # Additional geometry for this truss type
    overhang_length: float
    roof_pitch_deg: float

    # Computed properties
    roof_pitch: float

    @property
    @abstractmethod
    def type(self) -> str:
        return "[Generic] Roof Truss"

    def __init__(
        self,
        width: float,
        roof_pitch_deg: float,
        overhang_length: float = 0.0,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        self.roof_pitch_deg = roof_pitch_deg
        self.roof_pitch = np.radians(roof_pitch_deg)
        height = (width / 2) * np.tan(self.roof_pitch)
        self.overhang_length = overhang_length
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

    @abstractmethod
    def define_nodes(self) -> None:
        pass

    @abstractmethod
    def define_connectivity(self) -> None:
        pass

    def define_supports(self) -> None:
        bottom_left = 0
        bottom_right = max(self.bottom_chord_node_ids[0])
        self.support_definitions[bottom_left] = (
            self.supports_type if self.supports_type != "simple" else "pinned"
        )
        self.support_definitions[bottom_right] = (
            self.supports_type if self.supports_type != "simple" else "roller"
        )
