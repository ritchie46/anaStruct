from typing import Dict, Literal

AxisNumber = Literal[1, 2, 3]
Dimension = Literal["x", "y", "z", "both"]
ElementType = Literal["general", "truss"]
LoadDirection = Literal["element", "x", "y", "parallel", "perpendicular", "angle"]
MpType = Dict[Literal[1, 2], float]
OrientAxis = Literal["y", "z"]
Spring = Dict[Literal[1, 2], float]
SupportDirection = Literal["x", "y", "1", "2", 1, 2]
