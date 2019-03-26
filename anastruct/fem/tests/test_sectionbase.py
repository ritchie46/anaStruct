import unittest
from anastruct.sectionbase.sectionbase import section_base_proxy


class SimpleUnitTest(unittest.TestCase):

    def test_HE100B_EU(self):
        # HEB 100 B has mass=20.4kg/m Ax=26cm2 Iy=449.5cm4 Iz=167.3cm4

        section_base_proxy().set_unit_system(lenght_unit='m', mass_unit='kg', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / 26e-4, 1, places=2)
        self.assertAlmostEqual(param['mass'] / 20.4, 1, places=2)
        self.assertAlmostEqual(param['Iy'] / 449.5e-8, 1, places=2)
        self.assertAlmostEqual(param['Iz'] / 167.3e-8, 1, places=2)
        self.assertAlmostEqual(param['swdl'] / 20.4e1, 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='cm', mass_unit='kg', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / 26, 1, places=2)
        self.assertAlmostEqual(param['mass'] / 0.204, 1, places=2)
        self.assertAlmostEqual(param['Iy'] / 449.5, 1, places=2)
        self.assertAlmostEqual(param['Iz'] / 167.3, 1, places=2)
        self.assertAlmostEqual(param['swdl'] / 0.204e1, 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='mm', mass_unit='kg', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / 26e2, 1, places=2)
        self.assertAlmostEqual(param['mass'] / 0.0204, 1, places=2)
        self.assertAlmostEqual(param['Iy'] / 449.5e4, 1, places=2)
        self.assertAlmostEqual(param['Iz'] / 167.3e4, 1, places=2)
        self.assertAlmostEqual(param['swdl'] / 0.0204e1, 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='ft', mass_unit='kg', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / (26e-4 / (0.3048 ** 2)), 1, places=2)
        self.assertAlmostEqual(param['mass'] / (20.4 * 0.3048), 1, places=2)
        self.assertAlmostEqual(param['Iy'] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['Iz'] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['swdl'] / (20.4 * 0.3048 * 10), 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='inch', mass_unit='kg', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / (26e-4 / (0.0254 ** 2)), 1, places=2)
        self.assertAlmostEqual(param['mass'] / (20.4 * 0.0254), 1, places=2)
        self.assertAlmostEqual(param['Iy'] / (449.5e-8 / (0.0254) ** 4), 1, places=2)
        self.assertAlmostEqual(param['Iz'] / (167.3e-8 / (0.0254) ** 4), 1, places=2)
        self.assertAlmostEqual(param['swdl'] / (20.4 * 0.0254 * 10), 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='ft', mass_unit='lb', force_unit='N')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / (26e-4 / (0.3048 ** 2)), 1, places=2)
        self.assertAlmostEqual(param['mass'] / (20.4 * 0.3048 / 0.454), 1, places=2)
        self.assertAlmostEqual(param['Iy'] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['Iz'] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['swdl'] / (20.4 * 0.3048 * 10), 1, places=2)

        section_base_proxy().set_unit_system(lenght_unit='ft', mass_unit='lb', force_unit='lbf')
        param = section_base_proxy().get_sectionparameters('HE 100 B')
        self.assertAlmostEqual(param['Ax'] / (26e-4 / (0.3048 ** 2)), 1, places=2)
        self.assertAlmostEqual(param['mass'] / (20.4 * 0.3048 / 0.454), 1, places=2)
        self.assertAlmostEqual(param['Iy'] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['Iz'] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param['swdl'] / (20.4 * 0.3048 * 10 / 4.448), 1, places=2)

    def test_W12X65_US(self):
        # W12X65 has mass=65[lb/ft] Ax=19.1[inch2] Iy=533[inch4] Iz=174[inch4]
        section_base_proxy().set_databasename('US')
        section_base_proxy().set_unit_system(lenght_unit='inch', mass_unit='lb', force_unit='lbf')
        param = section_base_proxy().get_sectionparameters('W12X65')
        self.assertAlmostEqual(param['Ax'] / 19.1, 1, places=2)
        self.assertAlmostEqual(param['mass'] / (65 / 12), 1, places=2)
        self.assertAlmostEqual(param['Iy'] / 533, 1, places=2)
        self.assertAlmostEqual(param['Iz'] / 174, 1, places=2)
        self.assertAlmostEqual(param['swdl'] / (65 / 12), 1, places=2)


if __name__ == "__main__":
    unittest.main()
