from functools import lru_cache

@lru_cache(32000)
def det_moment(kl, kr, q, x, EI, L):
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
    return EI*(-L**3*kl*q*(6*EI + L*kr)/(12*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) +
               L*q*x*(12*EI**2 + 5*EI*L*kl + 3*EI*L*kr +
                      L**2*kl*kr)/(2*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) - q*x**2/(2*EI))


@lru_cache(32000)
def det_shear(kl, kr, q, x, EI, L):
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
    return EI*(L*q*(12*EI**2 + 5*EI*L*kl + 3*EI*L*kr + L**2*kl*kr) /
               (2*EI*(12*EI**2 + 4*EI*L*kl + 4*EI*L*kr + L**2*kl*kr)) - q*x/EI)


