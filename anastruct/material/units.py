"""
Converge profile units.

profile units:
* N
* mm
"""


def to_kN(v: float) -> float:
    return v / 1e3


def to_kNm2(v: float) -> float:
    return v * 1e-9
