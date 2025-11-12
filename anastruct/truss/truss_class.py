from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Union

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
    top_chord_node_ids: list[int] = []
    bottom_chord_node_ids: list[int] = []
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
        # Bottom chord elements
        for i, j in zip(
            self.bottom_chord_node_ids[:-1], self.bottom_chord_node_ids[1:]
        ):
            self.system.add_element(
                location=(self.nodes[i], self.nodes[j]),
                EA=self.bottom_chord_section["EA"],
                EI=self.bottom_chord_section["EI"],
                g=self.bottom_chord_section["g"],
                spring=None if self.bottom_chord_continuous else {1: 0.0, 2: 0.0},
            )

        # Top chord elements
        for i, j in zip(self.top_chord_node_ids[:-1], self.top_chord_node_ids[1:]):
            self.system.add_element(
                location=(self.nodes[i], self.nodes[j]),
                EA=self.top_chord_section["EA"],
                EI=self.top_chord_section["EI"],
                g=self.top_chord_section["g"],
                spring=None if self.top_chord_continous else {1: 0.0, 2: 0.0},
            )

        # Web diagonal elements
        for i, j in self.web_node_pairs:
            self.system.add_element(
                location=(self.nodes[i], self.nodes[j]),
                EA=self.web_section["EA"],
                EI=self.web_section["EI"],
                g=self.web_section["g"],
                spring={1: 0.0, 2: 0.0},
            )

        # Web vertical elements
        for i, j in self.web_verticals_node_pairs:
            self.system.add_element(
                location=(self.nodes[i], self.nodes[j]),
                EA=self.web_verticals_section["EA"],
                EI=self.web_verticals_section["EI"],
                g=self.web_verticals_section["g"],
                spring={1: 0.0, 2: 0.0},
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
