# StructuralEngineering 2D Frames and Trusses
[![Build Status](https://travis-ci.org/ritchie46/structural_engineering.svg?branch=master)](https://travis-ci.org/ritchie46/structural_engineering)

Compute 2D Frames and trusses for slender structures. Determine the bending moment, shear force, normal force and displacements.

## Note!

I am currently updating quite frequently, adding support for non linear nodes, better plotting options, numerical results and
unit tests. The syntax isn't backward compatible. When I think I'll keep the current syntax, I will update the blog
posts.


### Take a look at my blog for some examples.

* __[code examples!](https://ritchievink.com/blog/2017/01/12/python-1d-fem-example-1/)__

## Installation

For the 'old' release candidate:
```
$ pip install StructuralEngineering
```

For the actively developed version:
```
$ pip install git+https://github.com/ritchie46/structural_engineering.git
```

## 2D FEM Frames and Trusses
![](images/rand/structure.png)

## Development version

* trusses :heavy_check_mark:
* beams :heavy_check_mark:
* moment lines :heavy_check_mark:
* axial force lines :heavy_check_mark:
* shear force lines :heavy_check_mark:
* displacement lines :heavy_check_mark:
* hinged supports :heavy_check_mark:
* fixed supports :heavy_check_mark:
* spring supports :heavy_check_mark:
* q-load in elements direction :heavy_check_mark:
* point loads in global x, y directions on nodes :heavy_check_mark:
* dead load :heavy_check_mark:
* q-loads in global y direction :heavy_check_mark:
* hinged elements :heavy_check_mark:
* rotational springs :heavy_check_mark:
* non-linear nodes :heavy_check_mark:
* geometrical non linearity :heavy_multiplication_x:

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
system.add_support_fixed(node_id=1)
# Add a rotational spring at node 4.
system.add_support_spring(node_id=4, translation=3, K=4000)

# Add loads.
system.point_load(Fx=30, node_id=2)
system.q_load(q=10, elementID=2)

system.show_structure()
system.solve()
```
