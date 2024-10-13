
from anastruct.fem.system import SystemElements
ss=SystemElements()

# Add an element to the system (Example: Beam element between two nodes)
ss.add_element(location=[[0, 0], [5, 0]])  # Simple beam element


ss.get_stiffness_matrix(1)  # Element ID 1