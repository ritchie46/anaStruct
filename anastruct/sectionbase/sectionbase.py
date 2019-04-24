import os
import xml.etree.ElementTree as ET
from anastruct.sectionbase import units as u


class SectionBase:
    """
    Code is extracted from StruPy project.
    """
    available_database_names = ['EU', 'US', 'UK']

    def __init__(self):
        self.current_length_unit = None
        self.current_mass_unit = None
        self.current_force_unit = None
        self.current_database = None
        self.xml_length_unit = None
        self.xml_area_unit = None
        self.xml_weight_unit = None
        self.xml_self_weight_dead_load = None
        self.root = None  # xml root

        self.set_unit_system()

    def set_unit_system(self, length='m', mass_unit='kg', force_unit='N'):
        self.current_length_unit = u.l_dict[length]
        self.current_mass_unit = u.m_dict[mass_unit]
        self.current_force_unit = u.f_dict[force_unit]

    def set_database_name(self, basename):
        if basename == 'EU' or basename == 'UK':
            self.xml_length_unit = u.m
            self.xml_area_unit = u.m
            self.xml_weight_unit = u.kg
            self.xml_self_weight_dead_load = 10. * u.N
            if basename == 'EU':
                self.current_database = 'sectionbase_EuropeanSectionDatabase.xml'
            else:
                self.current_database = 'sectionbase_BritishSectionDatabase.xml'
        elif basename == 'US':
            self.current_database = 'sectionbase_AmericanSectionDatabase.xml'
            self.xml_length_unit = u.ft
            self.xml_area_unit = u.inch
            self.xml_weight_unit = u.lb
            self.xml_self_weight_dead_load = u.lbf

        self.load_data_from_xml()

    def load_data_from_xml(self):
        self.root = ET.parse(os.path.join(os.path.dirname(__file__), 'data', self.current_database)).getroot()

    def get_section_parameters(self, section_name):
        if self.root is None:
            self.set_database_name('EU')
            self.load_data_from_xml()

        element = dict(
            self.root.findall("./sectionlist/sectionlist_item[@sectionname='{}']".format(section_name))[0].items()
        )
        element['swdl'] = element['mass']
        element = self.convert_units(element)
        return element

    def convert_units(self, element):
        lu = self.xml_length_unit / self.current_length_unit  # long unit
        sdu = self.xml_area_unit / self.current_length_unit  # sect dim unit
        wu = self.xml_weight_unit / self.current_mass_unit  # weight unit
        self_weight_dead_load = self.xml_self_weight_dead_load / self.current_force_unit  # self weight dead load unit

        element['mass'] = float(element['mass']) * wu / lu
        element['Ax'] = float(element['Ax']) * sdu ** 2
        element['Iy'] = float(element['Iy']) * sdu ** 4
        element['Iz'] = float(element['Iz']) * sdu ** 4
        element['swdl'] = float(element['swdl']) * self_weight_dead_load / lu
        return element


section_base = SectionBase()
