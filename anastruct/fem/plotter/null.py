from anastruct.fem.plotter.values import PlottingValues


class Plotter(PlottingValues):
    def __init__(self, system, mesh):
        super(Plotter, self).__init__(system, mesh)
        self.system = system
        self.one_fig = None
        self.max_q = 0
        self.max_system_point_load = 0
