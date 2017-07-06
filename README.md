# StructuralEngineering 2D Frames and Trusses
Solve 2D Frames and Trusses calculations. Determine the bending moment, shear force, normal force and displacements.

### Take a look at my blog for some examples.

* __[code examples!](https://ritchievink.com/blog/2017/01/12/python-1d-fem-example-1/)__

## Installation

For the latest version download this repository and unzip the file. Open the command box in the unzipped file location and run:
```
$ python setup.py install
```

## 2D FEM Frames and Trusses
![](images/rand/structure.png)


```python
# if using ipython notebook
%matplotlib inline

import StructuralEngineering.FEM.system as se

# Create a new system object.
system = se.SystemElements()

# Add beams to the system. Positive z-axis is down, positive x-axis is the right.
system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)

# Add supports.
system.add_support_fixed(nodeID=1)
# Add a rotational spring at node 4.
system.add_support_spring(nodeID=4, translation=3, K=4000)

# Add loads.
system.point_load(Fx=30, nodeID=2)
system.q_load(q=10, elementID=2)

system.show_structure()
system.solve()
```
