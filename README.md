# StructuralEngineering 2D Frames and Trusses (Raamwerken en vakwerken)
This is a small package for python 3. With it you can solve 2D Frames and Trusses calculations. Determine the bending moment, shear force, normal force and the displacements.

### Don't feel like coding?  
Try it online: [www.laizen.nl](http://www.laizen.nl)

### See the wiki for some examples.

* __[Read the docs!](https://github.com/ritchie46/structural_engineering/wiki)__

## Installation

For the latest version download this repository and unzip the file. Open the command box in the unzipped file location and run:
```
$ python setup.py install
```

## 2D FEM Frames and Trusses
![](images/rand/structure.png)

M-kappa of a concrete cross section:

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
