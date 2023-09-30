import os
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from anastruct.sectionbase import units


class SectionBase:
    """
    Code is extracted from StruPy project.
    """

    available_database_names = ["EU", "US", "UK"]

    def __init__(self) -> None:
        self.current_length_unit: Optional[float] = None
        self.current_mass_unit: Optional[float] = None
        self.current_force_unit: Optional[float] = None
        self.current_database: Optional[str] = None
        self.xml_length_unit: Optional[float] = None
        self.xml_area_unit: Optional[float] = None
        self.xml_weight_unit: Optional[float] = None
        self.xml_self_weight_dead_load: Optional[float] = None
        self._root: Optional[ElementTree.Element] = None  # xml root

        self.set_unit_system()

    @property
    def root(self) -> ElementTree.Element:
        if self._root is None:
            self.set_database_name("EU")
            self.load_data_from_xml()
        assert self._root is not None
        return self._root

    @property
    def available_sections(self) -> list:
        return list(
            map(
                lambda el: el.attrib["sectionname"],
                self.root.findall("./sectionlist/sectionlist_item"),
            )
        )

    @property
    def available_units(self) -> Dict[str, List[str]]:
        return {
            "length": list(units.l_dict.keys()),
            "mass": list(units.m_dict.keys()),
            "force": list(units.f_dict.keys()),
        }

    def set_unit_system(
        self, length: str = "m", mass_unit: str = "kg", force_unit: str = "N"
    ) -> None:
        self.current_length_unit = units.l_dict[length]
        self.current_mass_unit = units.m_dict[mass_unit]
        self.current_force_unit = units.f_dict[force_unit]

    def set_database_name(self, basename: str) -> None:
        if basename in ("EU", "UK"):
            self.xml_length_unit = units.m
            self.xml_area_unit = units.m
            self.xml_weight_unit = units.kg
            self.xml_self_weight_dead_load = 10.0 * units.N
            if basename == "EU":
                self.current_database = "sectionbase_EuropeanSectionDatabase.xml"
            else:
                self.current_database = "sectionbase_BritishSectionDatabase.xml"
        elif basename == "US":
            self.current_database = "sectionbase_AmericanSectionDatabase.xml"
            self.xml_length_unit = units.ft
            self.xml_area_unit = units.inch
            self.xml_weight_unit = units.lb
            self.xml_self_weight_dead_load = units.lbf

        self.load_data_from_xml()

    def load_data_from_xml(self) -> None:
        assert self.current_database is not None
        self._root = ElementTree.parse(
            os.path.join(os.path.dirname(__file__), "data", self.current_database)
        ).getroot()

    def get_section_parameters(self, section_name: str) -> dict:
        if self.root is None:
            self.set_database_name("EU")
            self.load_data_from_xml()

        element = dict(
            self.root.findall(
                f"./sectionlist/sectionlist_item[@sectionname='{section_name}']"
            )[0].items()
        )
        element["swdl"] = element["mass"]
        element = self.convert_units(element)
        return element

    def convert_units(self, element: Dict[str, Any]) -> Dict[str, Any]:
        assert self.current_length_unit is not None
        assert self.current_mass_unit is not None
        assert self.current_force_unit is not None
        assert self.xml_length_unit is not None
        assert self.xml_area_unit is not None
        assert self.xml_weight_unit is not None
        assert self.xml_self_weight_dead_load is not None

        lu = self.xml_length_unit / self.current_length_unit  # long unit
        sdu = self.xml_area_unit / self.current_length_unit  # sect dim unit
        wu = self.xml_weight_unit / self.current_mass_unit  # weight unit
        self_weight_dead_load = (
            self.xml_self_weight_dead_load / self.current_force_unit
        )  # self weight dead load unit

        element["mass"] = float(element["mass"]) * wu / lu
        element["Ax"] = float(element["Ax"]) * sdu**2
        element["Iy"] = float(element["Iy"]) * sdu**4
        element["Iz"] = float(element["Iz"]) * sdu**4
        element["swdl"] = float(element["swdl"]) * self_weight_dead_load / lu
        return element


section_base = SectionBase()
