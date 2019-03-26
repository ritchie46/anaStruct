import os
import copy
import xml.etree.ElementTree as ET
from anastruct.sectionbase import units as u


class SectionBase:

    def __init__(self, basename='EU'):
        self.out_lu = None
        self.out_mu = None
        self.out_fu = None
        self.__database = None
        self.xml_lu = None
        self.xml_sdu = None
        self.xml_wu = None
        self.xml_swdlu = None
        self.set_databasename(basename)
        self.set_unit_system()
        self.__load_data_from_xml()

    def set_unit_system(self, lenght_unit='m', mass_unit='kg', force_unit='N'):
        self.out_lu = u.l_dict[lenght_unit]
        self.out_mu = u.m_dict[mass_unit]
        self.out_fu = u.f_dict[force_unit]

    def set_databasename(self, basename):
        if basename == 'EU':
            self.__database = 'sectionbase_EuropeanSectionDatabase.xml'
            self.xml_lu = u.m
            self.xml_sdu = u.m
            self.xml_wu = u.kg
            self.xml_swdlu = 10. * u.N
            self.__load_data_from_xml()
        elif basename == 'US':
            self.__database = 'sectionbase_AmericanSectionDatabase.xml'
            self.xml_lu = u.ft
            self.xml_sdu = u.inch
            self.xml_wu = u.lb
            self.xml_swdlu = u.lbf
            self.__load_data_from_xml()
        elif basename == 'UK':
            self.__database = 'sectionbase_BritishSectionDatabase.xml'
            self.xml_lu = u.m
            self.xml_sdu = u.m
            self.xml_wu = u.kg
            self.xml_swdlu = 10 * u.N
            self.__load_data_from_xml()
        else:
            pass

    def __load_data_from_xml(self):
        self.__package_dir = os.path.split(__file__)[0]
        self.__xmlbase_path = os.path.join(self.__package_dir, self.__database)
        self.__tree = ET.parse(self.__xmlbase_path)
        self.__root = self.__tree.getroot()

    def __get_sectiondictwithparam(self, sectionname):
        sectiondictwithparam = {}
        for item in self.__root.iter('sectionlist_item'):
            if item.attrib['sectionname'] == sectionname:
                sectiondictwithparam = copy.deepcopy(item.attrib)
                sectiondictwithparam['swdl'] = sectiondictwithparam['mass']
                sectiondictwithparam = self.__sectionparameters_unitaplly(sectiondictwithparam)
        sectiondictwithparam['Wy'] = \
            sectiondictwithparam['Iy'] / max(sectiondictwithparam['vz'], sectiondictwithparam['vpz'])
        sectiondictwithparam['Wz'] = \
            sectiondictwithparam['Iz'] / max(sectiondictwithparam['vy'], sectiondictwithparam['vpy'])
        return sectiondictwithparam

    def __sectionparameters_unitaplly(self, param):
        lu = self.xml_lu / self.out_lu  # long unit
        sdu = self.xml_sdu / self.out_lu  # sectdim unit
        wu = self.xml_wu / self.out_mu  # weight unit
        swdlu = self.xml_swdlu / self.out_fu  # self weightdead load unit
        # --
        param['mass'] = float(param['mass']) * wu / lu
        param['surf'] = float(param['surf']) * sdu ** 2 / lu
        param['h'] = float(param['h']) * sdu
        param['b'] = float(param['b']) * sdu
        param['ea'] = float(param['ea']) * sdu
        param['es'] = float(param['es']) * sdu
        param['ra'] = float(param['ra']) * sdu
        param['rs'] = float(param['rs']) * sdu
        param['gap'] = float(param['gap']) * sdu
        param['Ax'] = float(param['Ax']) * sdu ** 2
        param['Ay'] = float(param['Ay']) * sdu ** 2
        param['Az'] = float(param['Az']) * sdu ** 2
        param['Ix'] = float(param['Ix']) * sdu ** 4
        param['Iy'] = float(param['Iy']) * sdu ** 4
        param['Iz'] = float(param['Iz']) * sdu ** 4
        param['Iomega'] = float(param['Iomega']) * sdu ** 6
        param['vy'] = float(param['vy']) * sdu
        param['vpy'] = float(param['vpy']) * sdu
        param['vz'] = float(param['vz']) * sdu
        param['vpz'] = float(param['vpz']) * sdu
        param['Wply'] = float(param['Wply']) * sdu ** 3
        param['Wplz'] = float(param['Wplz']) * sdu ** 3
        param['Wtors'] = float(param['Wtors']) * sdu ** 3
        param['Ayv'] = float(param['Ayv']) * sdu ** 2
        param['Azv'] = float(param['Azv']) * sdu ** 2
        param['swdl'] = float(param['swdl']) * swdlu / lu
        param['gamma'] = float(param['gamma'])
        return param

    def get_available_databasenames(self):
        return ['EU', 'US', 'UK']

    def get_available_unit_system(self):
        return [u.l_dict.keys(), u.m_dict.keys(), u.f_dict.keys()]

    def get_database_name(self):
        name = []
        for item in self.__root.iter('baseinformation'):
            name = item.attrib
        return name['basetitle']

    def get_sectionparameters(self, sectname='IPE 270'):
        return self.__get_sectiondictwithparam(sectname)

    def get_database_sectiontypes(self):
        sectiontypes = []
        for item in self.__root.iter('sectiontype_item'):
            sectiontypes.append(item.attrib['figure'])
        return sectiontypes

    def get_database_sectiontypesdescription(self):
        description = {}
        for item in self.__root.iter('sectiontype_item'):
            description[item.attrib['figure']] = item.attrib['description']
        return description

    def get_database_sectionlist(self):
        sectionlist = []
        for item in self.__root.iter('sectionlist_item'):
            sectionlist.append(item.attrib['sectionname'])
        return sectionlist

    def get_database_sectionlistwithtype(self, secttype='IPE'):
        sectionlistwithtype = []
        for item in self.__root.iter('sectionlist_item'):
            if item.attrib['figure'] == secttype:
                sectionlistwithtype.append(item.attrib['sectionname'])
        return sectionlistwithtype

    def get_parameters_description(self):
        descriptiondir = []
        for item in self.__root.iter('parameterdescription'):
            descriptiondir = item.attrib
        descriptiondir['Wy'] = 'Elastic section modulus about Y axis'
        descriptiondir['Wz'] = 'Elastic section modulus about Z axis'
        descriptiondir['swdl'] = 'Self weight dead load'
        descriptiondir['figuretype'] = 'General type of section figure'
        return descriptiondir
