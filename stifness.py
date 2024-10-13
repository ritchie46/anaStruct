
from anastruct.fem import system as se
ss=se.SystemElements()

# Add an element to the system (Example: Beam element between two nodes)
ss.add_element(location=[[0, 0], [5, 0]])  # Simple beam element

ss.add_element(location=[[0, 0], [5, 0]])

ss.add_element(location=[[0, 4], [8, 0]])

ss=ss.get_stiffness_matrix(element_id=4)  # Element ID 1

print(ss)

