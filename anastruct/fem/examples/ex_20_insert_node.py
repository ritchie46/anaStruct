from anastruct.fem.system import SystemElements

ss = SystemElements()

ss.add_element([10, 0])
ss.insert_node(1, factor=0.3)
ss.insert_node(2, [5, 5])

if __name__ == "__main__":
    ss.show_structure(values_only=True)
