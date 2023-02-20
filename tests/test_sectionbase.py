import unittest
from anastruct.sectionbase import section_base
from anastruct.fem import system


class SimpleUnitTest(unittest.TestCase):
    def setUp(self):
        section_base.set_database_name("EU")
        section_base.set_unit_system(length="m", mass_unit="kg", force_unit="N")

    def test_HE100B_EU(self):
        # HEB 100 B has mass=20.4kg/m Ax=26.036cm2 Iy=449.5cm4 Iz=167.3cm4

        section_base.set_unit_system(length="m", mass_unit="kg", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / 26e-4, 1, places=2)
        self.assertAlmostEqual(param["mass"] / 20.4, 1, places=2)
        self.assertAlmostEqual(param["Iy"] / 449.5e-8, 1, places=2)
        self.assertAlmostEqual(param["Iz"] / 167.3e-8, 1, places=2)
        self.assertAlmostEqual(param["swdl"] / 20.4e1, 1, places=2)

        section_base.set_unit_system(length="cm", mass_unit="kg", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / 26, 1, places=2)
        self.assertAlmostEqual(param["mass"] / 0.204, 1, places=2)
        self.assertAlmostEqual(param["Iy"] / 449.5, 1, places=2)
        self.assertAlmostEqual(param["Iz"] / 167.3, 1, places=2)
        self.assertAlmostEqual(param["swdl"] / 0.204e1, 1, places=2)

        section_base.set_unit_system(length="mm", mass_unit="kg", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / 26e2, 1, places=2)
        self.assertAlmostEqual(param["mass"] / 0.0204, 1, places=2)
        self.assertAlmostEqual(param["Iy"] / 449.5e4, 1, places=2)
        self.assertAlmostEqual(param["Iz"] / 167.3e4, 1, places=2)
        self.assertAlmostEqual(param["swdl"] / 0.0204e1, 1, places=2)

        section_base.set_unit_system(length="ft", mass_unit="kg", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / (26e-4 / (0.3048**2)), 1, places=2)
        self.assertAlmostEqual(param["mass"] / (20.4 * 0.3048), 1, places=2)
        self.assertAlmostEqual(param["Iy"] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param["Iz"] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param["swdl"] / (20.4 * 0.3048 * 10), 1, places=2)

        section_base.set_unit_system(length="inch", mass_unit="kg", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / (26e-4 / (0.0254**2)), 1, places=2)
        self.assertAlmostEqual(param["mass"] / (20.4 * 0.0254), 1, places=2)
        self.assertAlmostEqual(param["Iy"] / (449.5e-8 / (0.0254) ** 4), 1, places=2)
        self.assertAlmostEqual(param["Iz"] / (167.3e-8 / (0.0254) ** 4), 1, places=2)
        self.assertAlmostEqual(param["swdl"] / (20.4 * 0.0254 * 10), 1, places=2)

        section_base.set_unit_system(length="ft", mass_unit="lb", force_unit="N")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / (26e-4 / (0.3048**2)), 1, places=2)
        self.assertAlmostEqual(param["mass"] / (20.4 * 0.3048 / 0.454), 1, places=2)
        self.assertAlmostEqual(param["Iy"] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param["Iz"] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param["swdl"] / (20.4 * 0.3048 * 10), 1, places=2)

        section_base.set_unit_system(length="ft", mass_unit="lb", force_unit="lbf")
        param = section_base.get_section_parameters("HE 100 B")
        self.assertAlmostEqual(param["Ax"] / (26e-4 / (0.3048**2)), 1, places=2)
        self.assertAlmostEqual(param["mass"] / (20.4 * 0.3048 / 0.454), 1, places=2)
        self.assertAlmostEqual(param["Iy"] / (449.5e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(param["Iz"] / (167.3e-8 / (0.3048) ** 4), 1, places=2)
        self.assertAlmostEqual(
            param["swdl"] / (20.4 * 0.3048 * 10 / 4.448), 1, places=2
        )

    def test_W12X65_US(self):
        # W12X65 has mass=65[lb/ft] Ax=19.1[inch2] Iy=533[inch4] Iz=174[inch4]
        section_base.set_database_name("US")
        section_base.set_unit_system(length="inch", mass_unit="lb", force_unit="lbf")
        param = section_base.get_section_parameters("W12X65")
        self.assertAlmostEqual(param["Ax"] / 19.1, 1, places=2)
        self.assertAlmostEqual(param["mass"] / (65 / 12), 1, places=2)
        self.assertAlmostEqual(param["Iy"] / 533, 1, places=2)
        self.assertAlmostEqual(param["Iz"] / 174, 1, places=2)
        self.assertAlmostEqual(param["swdl"] / (65 / 12), 1, places=2)

    def test_sectionbase_steel_section_deflection(self):
        ss = system.SystemElements()
        ss.add_multiple_elements(
            [[0, 0], [6, 0]], n=2, steelsection="IPE 300", E=210e9, sw=False, orient="y"
        )
        ss.add_support_hinged(1)
        ss.add_support_hinged(3)
        ss.point_load(2, Fy=-10000)
        ss.solve()
        self.assertAlmostEqual(-0.00256442, ss.get_node_displacements(2)["uy"])

    def test_sectionbase_steel_section_self_weight_reaction(self):
        ss = system.SystemElements()
        ss.add_element([[0, 0], [20, 0]], steelsection="IPE 300", sw=True)
        ss.add_support_hinged(1)
        ss.add_support_hinged(2)
        ss.solve()
        self.assertAlmostEqual(-4224.24, ss.reaction_forces[1].Fz)

    def test_rectangle_section_deflection(self):
        ss = system.SystemElements()
        ss.add_multiple_elements(
            [[0, 0], [2, 0]], n=2, h=0.05, b=0.16, E=210e9, sw=False, orient="y"
        )
        ss.add_support_hinged(1)
        ss.add_support_hinged(3)
        ss.point_load(2, Fy=-1000)
        ss.solve()
        self.assertAlmostEqual(-0.00047619, ss.get_node_displacements(2)["uy"])

    def test_circle_section_deflection(self):
        ss = system.SystemElements()
        ss.add_multiple_elements([[0, 0], [2, 0]], n=2, d=0.07, E=210e9, sw=False)
        ss.add_support_hinged(1)
        ss.add_support_hinged(3)
        ss.point_load(2, Fy=-1000)
        ss.solve()
        self.assertAlmostEqual(-0.00067339, ss.get_node_displacements(2)["uy"])

    def test_rectangle_section_self_weight_reaction(self):
        ss = system.SystemElements()
        ss.add_element([[0, 0], [2, 0]], h=0.1, b=0.1, E=210e9, sw=True, gamma=10000)
        ss.add_support_hinged(1)
        ss.add_support_hinged(2)
        ss.solve()
        self.assertAlmostEqual(-100, ss.reaction_forces[1].Fz)

    def test_circle_section_self_weight_reaction(self):
        ss = system.SystemElements()
        ss.add_element([[0, 0], [2, 0]], d=0.2, E=210e9, sw=True, gamma=10000)
        ss.add_support_hinged(1)
        ss.add_support_hinged(2)
        ss.solve()
        self.assertAlmostEqual(-314.15926535, ss.reaction_forces[1].Fz)

    def test_show_available_sections(self):
        sections = section_base.available_sections
        self.assertEqual(sections[0], "CAE 100x10")
        self.assertEqual(len(sections), 2145)

    def test_available_units(self):
        self.assertEqual(section_base.available_units["length"][0], "m")


if __name__ == "__main__":
    unittest.main()
