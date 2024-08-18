from functools import lru_cache


@lru_cache(32000)
def det_moment(
    kl: float, kr: float, qi: float, q: float, x: float, EI: float, L: float
) -> float:
    """
    See notebook in: anastruct/fem/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI * (
        -(L**3)
        * kl
        * (14 * EI * q + 16 * EI * qi + 2 * L * kr * q + 3 * L * kr * qi)
        / (60 * EI * (12 * EI**2 + 4 * EI * L * kl + 4 * EI * L * kr + L**2 * kl * kr))
        - L
        * x
        * (
            -2 * L * (q + 4 * qi) * (EI * kr + kl * (EI + L * kr))
            + 5
            * (2 * EI + L * kl)
            * (4 * EI * q + 8 * EI * qi + L * kr * q + 3 * L * kr * qi)
        )
        / (
            20
            * EI
            * (
                2 * EI * L * kr
                - 6 * EI * (2 * EI + L * kr)
                + 2 * L * kl * (EI + L * kr)
                - 3 * L * kl * (2 * EI + L * kr)
            )
        )
        - qi * x**2 / (2 * EI)
        - x**3 * (q - qi) / (6 * EI * L)
    )


@lru_cache(32000)
def det_shear(
    kl: float, kr: float, qi: float, q: float, x: float, EI: float, L: float
) -> float:
    """
    See notebook in: anastruct/fem/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI * (
        -L
        * (
            -2 * L * (q + 4 * qi) * (EI * kr + kl * (EI + L * kr))
            + 5
            * (2 * EI + L * kl)
            * (4 * EI * q + 8 * EI * qi + L * kr * q + 3 * L * kr * qi)
        )
        / (
            20
            * EI
            * (
                2 * EI * L * kr
                - 6 * EI * (2 * EI + L * kr)
                + 2 * L * kl * (EI + L * kr)
                - 3 * L * kl * (2 * EI + L * kr)
            )
        )
        - qi * x / EI
        - x**2 * (q - qi) / (2 * EI * L)
    )


@lru_cache(32000)
def det_axial(qi: float, q: float, x: float, EA: float, L: float) -> float:
    """
    See notebook in: anastruct/fem/background/distributed_ax_force.ipynb

    :param q: (flt)
    :param x: (flt) Location of the axial force
    :param EA: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EA * (
        x * (-L * qi / 2 + x * (-q + qi) / 3) / (EA * L)
        + (L**2 * (q + 2 * qi) / 6 - L * qi * x / 2 + x**2 * (-q + qi) / 6) / (EA * L)
    )
