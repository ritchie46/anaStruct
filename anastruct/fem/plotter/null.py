from typing import TYPE_CHECKING, Optional

from anastruct.fem.plotter.values import PlottingValues

if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore
    from matplotlib.figure import Figure  # type: ignore

    from anastruct.fem.system import SystemElements


class Plotter(PlottingValues):
    def __init__(self, system: "SystemElements", mesh: int):
        super(Plotter, self).__init__(system, mesh)
        self.system: "SystemElements" = system
        self.one_fig: Optional["Axes"] = None
        self.max_q: float = 0
        self.max_qn: float = 0
        self.max_system_point_load: float = 0
        self.fig: Optional["Figure"] = None
