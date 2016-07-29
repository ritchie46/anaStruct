class SystemLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
        post_el = ElementLevel(self.system)
        post_el.node_results()

        count = 0
        for node in self.system.node_objects:
            for el in self.system.elements:
                if el.node_1.ID == node.ID:
                    self.system.node_objects[count] += el.node_1
                elif el.node_2.ID == node.ID:
                    self.system.node_objects[count] += el.node_2
            count += 1


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self):
        for el in self.system.elements:
            el.determine_node_results()
