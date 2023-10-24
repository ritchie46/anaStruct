"""
Converge profile units.

profile units:
* N
* mm
"""


def to_kN(v: float) -> float:
    """Convert N to kN

    Args:
        v (float): Value in N

    Returns:
        float: Value in kN
    """
    return v / 1e3


def to_kNm2(v: float) -> float:
    """Convert N/m^2 to kN/m^2

    Args:
        v (float): Value in N/m^2

    Returns:
        float: Value in kN/m^2
    """
    return v * 1e-9
