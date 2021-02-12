"""
Converge profile units.

profile units:
* N
* mm
"""


def to_kN(v):
    return v / 1e3


def to_kNm2(v):
    return v * 1e-9
