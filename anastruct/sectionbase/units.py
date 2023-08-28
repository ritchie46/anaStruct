from typing import Dict

m: float = 1.0
cm: float = 1.0e-2
mm: float = 1.0e-3
kg: float = 1.0
N: float = 1.0
kN: float = 1.0e3
# IMPERIAL
inch: float = 0.0254
ft: float = 12.0 * inch
lb: float = 0.4536
lbf: float = 4.448
kip: float = 1.0e3 * lbf

l_dict: Dict[str, float] = {"m": m, "cm": cm, "mm": mm, "ft": ft, "inch": inch}
m_dict: Dict[str, float] = {"kg": kg, "lb": lb}
f_dict: Dict[str, float] = {"N": N, "kN": kN, "lbf": lbf, "kip": kip}
