import math

import matplotlib.pyplot as plt
import numpy as np

from StructuralEngineering.cross_section import CrossSection, Epsilon


class MaterialConcrete:
    def __init__(self, fck, fctk, yield_strain=0.00175, max_strain=0.0035, gamma=1.5):
        self.nValues = 100
        self.yield_strain = yield_strain
        self.max_strain = max_strain
        self.fck = fck
        self.fctk = fctk
        self.gamma = gamma
        self.fcd = self.fck / self.gamma
        self.fctd = self.fctk / self.gamma
        self.E_fictious = self.fcd / self.yield_strain
        self.crack_strain = self.fctd / self.E_fictious
        self.strain_diagram = np.linspace(0, self.max_strain, self.nValues)
        self.stress_diagram = np.zeros(self.nValues)

    def det_bi_linear_diagram(self):
        for i in range(self.nValues):
            eps = self.strain_diagram[i]

            if eps < self.yield_strain:
                sig = self.fcd / self.yield_strain * eps
                self.stress_diagram[i] = sig
            else:
                self.stress_diagram[i] = self.fcd


class MaterialRebar:
    def __init__(self, fyk, gamma=1.15):
        self.E_modulus = 2e5
        self.fyk = fyk
        self.gamma = gamma
        self.fyd = self.fyk / self.gamma
        self.yieldStrain = self.fyd / self.E_modulus


class ReinforcedConcrete(CrossSection):
    def __init__(self, coordinate_list, materialConcrete, materialRebar):
        CrossSection.__init__(self, coordinate_list)
        self.concrete = materialConcrete
        self.materialRebar = materialRebar
        self.rebarAs = []
        self.rebar_d = []
        self.rebarHeightMinus_d = []
        self.moment = [0]
        self.kappa = [0]

    def add_rebar(self, n, diam, d):
        As = n * 0.25 * math.pi * diam**2
        self.rebarAs.append(As)
        self.rebar_d.append(d)
        self.rebarHeightMinus_d.append(self.height - d)

    def plot_M_Kappa(self):
        self.cracking_moment()
        self.yielding_moment()
        self.plastic_concrete_moment()
        self.crushing_moment()

        plt.plot(self.kappa, self.moment)
        plt.show()

    def cracking_moment(self):
        strain = Epsilon()
        strainAtTop = -0.001
        eps = strain.create_eps_array(strainAtTop, self.concrete.crack_strain, self.nValues)
        while 1:
            concrete_compression = self.sum_concrete_compression(eps)
            concrete_tension = self.sum_concrete_tension(eps)
            rebar = self.sum_rebar(eps)

            # if sum of all force equals zero -> break
            sol = self.sum_tensile_compression(concrete_compression, rebar, concrete_tension)
            if convergence_conditions(sol[0], sol[1]):
                moment = determine_moment(concrete_compression, rebar, self.rebarHeightMinus_d, concrete_tension)
                Xc = self.determine_Xc(eps)
                kappa = eps[0] / Xc

                self.moment.append(-moment)
                self.kappa.append(kappa)
                return moment

            else:
                factor = convergence(sol[0], sol[1])
                strainAtTop *= factor
                eps = strain.create_eps_array(strainAtTop, self.concrete.crack_strain, self.nValues)

    def yielding_moment(self):
        strain = Epsilon()
        strainAtTop = -0.001

        # determine most lower rebar
        d = max(self.rebar_d)

        totalStrainOver_d = - strainAtTop + self.materialRebar.yieldStrain
        totalStrainOverHeight = totalStrainOver_d / d * self.height
        strainAtBottom = totalStrainOverHeight + strainAtTop

        eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)
        while 1:
            concrete_compression = self.sum_concrete_compression(eps)
            rebar = self.sum_rebar(eps)

            # if sum of all force equals zero -> break
            sol = self.sum_tensile_compression(concrete_compression, rebar)
            if convergence_conditions(sol[0], sol[1]):
                moment = determine_moment(concrete_compression, rebar, self.rebarHeightMinus_d)
                Xc = self.determine_Xc(eps)
                kappa = eps[0] / Xc

                self.moment.append(-moment)
                self.kappa.append(kappa)
                return moment

            else:
                factor = convergence(sol[0], sol[1])
                strainAtTop *= factor

                totalStrainOver_d = - strainAtTop + self.materialRebar.yieldStrain
                totalStrainOverHeight = totalStrainOver_d / d * self.height
                strainAtBottom = totalStrainOverHeight + strainAtTop

                eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)

    def plastic_concrete_moment(self):
        strain = Epsilon()
        strainAtTop = -self.concrete.yield_strain

        # starting strain for the iteration
        strainAtBottom = 0.002175

        eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)
        while 1:
            concrete_compression = self.sum_concrete_compression(eps)
            rebar = self.sum_rebar(eps)

            # if sum of all force equals zero -> break
            sol = self.sum_tensile_compression(concrete_compression, rebar)
            if convergence_conditions(sol[0], sol[1]):
                moment = determine_moment(concrete_compression, rebar, self.rebarHeightMinus_d)
                Xc = self.determine_Xc(eps)
                kappa = eps[0] / Xc

                self.moment.append(-moment)
                self.kappa.append(kappa)
                return moment

            else:
                factor = convergence(sol[1], sol[0])
                strainAtBottom *= factor
                eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)

    def crushing_moment(self):
        strain = Epsilon()
        strainAtTop = -self.concrete.max_strain

        # starting strain for the iteration
        strainAtBottom = 0.002175

        eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)
        while 1:
            concrete_compression = self.sum_concrete_compression(eps)
            rebar = self.sum_rebar(eps)

            # if sum of all force equals zero -> break
            sol = self.sum_tensile_compression(concrete_compression, rebar)
            if convergence_conditions(sol[0], sol[1]):
                moment = determine_moment(concrete_compression, rebar, self.rebarHeightMinus_d)
                Xc = self.determine_Xc(eps)
                kappa = eps[0] / Xc

                self.moment.append(-moment)
                self.kappa.append(kappa)
                return moment

            else:
                factor = convergence(sol[1], sol[0])
                strainAtBottom *= factor
                eps = strain.create_eps_array(strainAtTop, strainAtBottom, self.nValues)

    def sum_concrete_compression(self, strain):
            sum_force = 0
            sum_f_z = 0
            for i in range(self.nValues):

                if strain[i] < 0:
                    index = find_closest_index(self.concrete.strain_diagram, -strain[i])
                    stress = -self.concrete.stress_diagram[index]
                    force = stress * self.width_array[i] * self.section_height
                    sum_force += force
                    # determine the center of force by sum(force * z) / sum(force)
                    sum_f_z += force * self.height_array[i]

            z = sum_f_z / sum_force
            return sum_force, z

    def sum_concrete_tension(self, strain):
        sum_force = 0
        sum_f_z = 0
        for i in range(self.nValues):
            if strain[i] > 0:
                force = strain[i] * self.concrete.E_fictious * self.width_array[i] * self.section_height
                sum_force += force
                # determine the center of force by sum(force * z) / sum(force)
                sum_f_z += force * self.height_array[i]

        z = sum_f_z / sum_force
        return sum_force, z

    def sum_rebar(self, strain):
        """
        :return: sum of rebar forces and force list
        """

        sum_rebar = 0
        forceList = []
        for i in range(len(self.rebar_d)):
            eps_at_d = interpolate_strain(
                strainArray=strain, height=self.height, rebar_d=self.rebar_d[i])
            if eps_at_d > self.materialRebar.yieldStrain:
                force = self.rebarAs[i] * self.materialRebar.fyd
            elif eps_at_d < - self.materialRebar.yieldStrain:
                force = self.rebarAs[i] * -self.materialRebar.fyd
            else:
                force = eps_at_d * self.materialRebar.E_modulus * self.rebarAs[i]
            sum_rebar += force
            forceList.append(force)

        return sum_rebar, forceList

    def sum_tensile_compression(self, concrete_compression, rebar, concrete_tension=(0,)):
        """
        :param concrete_compression: tuple:( force, z)
        :param rebar: tuple: (force, list)
        :param concrete_tension: tuple: (force, z)
        :return:
        """
        compression = concrete_compression[0]
        tension = concrete_tension[0]

        for i in rebar[1]:
            if i < 0:
                compression += i
            else:
                tension += i

        return compression, tension

    def determine_Xc(self, strain_list):
        sum_strain = -strain_list[0] + strain_list[-1]
        Xc = self.height / sum_strain * strain_list[0]
        return Xc


def determine_moment(concrete_compression, rebar, rebar_height_minus_d, concrete_tension=(0, 0)):
    moment = 0
    moment += concrete_compression[0] * concrete_compression[1]
    moment += concrete_tension[0] * concrete_tension[1]

    for i in range(len(rebar_height_minus_d)):
        moment += rebar[1][i] * rebar_height_minus_d[i]
    return moment


def convergence_conditions(leftHandSide, rightHandSide):
    ratio = abs(leftHandSide) / abs(rightHandSide)

    if 0.999 <= ratio <= 1.001:
        return True
    else:
        return False


def find_closest_index(array, searchFor):
    """
    Find the closest index in an ordered array
    :param array: array searched in
    :param searchFor: value that is searched
    :return: index of the closest match
    """
    if len(array) == 1:
        return 0

    # determine the direction of the iteration
    if array[1] < array[0] > searchFor:  # iteration from big to small

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return 0
        elif array[0] > searchFor and array[-1] > searchFor:
            return len(array) - 1

        for i in range(len(array)):
            if array[i] <= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                # search for the index with the smallest absolute difference
                if searchFor - valueAtIndex <= previousValue - searchFor:
                    return i
                else:
                    return i - 1

    else:
        # array[0] is smaller than searchFor
        # iteration from small to big

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return len(array) - 1
        elif array[0] > searchFor and array[-1] > searchFor:
            return 0

        for i in range(len(array)):
            if array[i] >= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                if valueAtIndex - searchFor <= searchFor - previousValue:
                    return i
                else:
                    return i - 1


def convergence(leftHandSide, rightHandSide):
    """
    Converting by adapting one value by a factor. The factor is determined by the ratio of the left hand side and
    the right hand side of the equation.

    Factor:
    ((Left / Right) - 1) / 3 + 1

    :param leftHandSide: Scalar
    :param rightHandSide: Scalar
    :return: Factor
    """
    ratio = abs(rightHandSide) / abs(leftHandSide)
    return (ratio - 1) / 3 + 1


def interpolate_strain(strainArray, height, rebar_d):
    """
    Determine the strain in the rebar.
    :param strainArray: Array of strains of the cross-section. array[0] = compression, array[-1] = tension
    :param height: Height of the cross-section
    :param rebar_d: distance of the center of the rebar and the top of the cross-section
    :return:
    """
    totalStrain = -strainArray[0] + strainArray[-1]

    # totalStrain_d = total strain at point d and strain at the top
    totalStrain_d = totalStrain / height * rebar_d

    # strain at point d
    strain_d = totalStrain_d + strainArray[0]
    return strain_d
