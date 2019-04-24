import os
import xml.etree.ElementTree as ET
from anastruct.sectionbase import units as u


class SectionBase:
    """
    Code is extracted from StruPy project.
    """
    available_database_names = ['EU', 'US', 'UK']

    def __init__(self, basename='EU'):
        self.out_lu = None
        self.out_mu = None
        self.out_fu = None
        self.current_database = None
        self.xml_lu = None
        self.xml_sdu = None
        self.xml_wu = None
        self.xml_self_weight_dead_load = None
        self.root = None  # xml root

        self.set_database_name(basename)
        self.set_unit_system()
        self.load_data_from_xml()

    def set_unit_system(self, lenght_unit='m', mass_unit='kg', force_unit='N'):
        self.out_lu = u.l_dict[lenght_unit]
        self.out_mu = u.m_dict[mass_unit]
        self.out_fu = u.f_dict[force_unit]

    def set_database_name(self, basename):
        if basename == 'EU' or basename == 'UK':
            self.xml_lu = u.m
            self.xml_sdu = u.m
            self.xml_wu = u.kg
            self.xml_self_weight_dead_load = 10. * u.N
            if basename == 'EU':
                self.current_database = 'sectionbase_EuropeanSectionDatabase.xml'
            else:
                self.current_database = 'sectionbase_BritishSectionDatabase.xml'
        elif basename == 'US':
            self.current_database = 'sectionbase_AmericanSectionDatabase.xml'
            self.xml_lu = u.ft
            self.xml_sdu = u.inch
            self.xml_wu = u.lb
            self.xml_self_weight_dead_load = u.lbf

        self.load_data_from_xml()

    def load_data_from_xml(self):
        self.root = ET.parse(os.path.join(os.path.dirname(__file__), 'data', self.current_database)).getroot()

    def get_section_parameters(self, section_name):
        element = dict(
            self.root.findall("./sectionlist/sectionlist_item[@sectionname='{}']".format(section_name))[0].items()
        )
        element['swdl'] = element['mass']
        element = self.convert_units(element)

        return element

    def convert_units(self, element):
        lu = self.xml_lu / self.out_lu  # long unit
        sdu = self.xml_sdu / self.out_lu  # sect dim unit
        wu = self.xml_wu / self.out_mu  # weight unit
        self_weight_dead_load = self.xml_self_weight_dead_load / self.out_fu  # self weight dead load unit

        element['mass'] = float(element['mass']) * wu / lu
        element['Ax'] = float(element['Ax']) * sdu ** 2
        element['Iy'] = float(element['Iy']) * sdu ** 4
        element['Iz'] = float(element['Iz']) * sdu ** 4
        element['swdl'] = float(element['swdl']) * self_weight_dead_load / lu
        return element


class SectionBaseProxy:
    """
    utility class to ensure data is not loaded when not needed.
    """
    data = None

    @classmethod
    def __call__(cls, *args, **kwargs):
        if cls.data is None:
            cls.data = SectionBase()
        return cls.data


section_base_proxy = SectionBaseProxy()