# anaStruct 2D Frames and Trusses
[![Build Status](https://travis-ci.org/ritchie46/anaStruct.svg?branch=master)](https://travis-ci.org/ritchie46/anaStruct)
[![Documentation Status](https://readthedocs.org/projects/anastruct/badge/?version=latest)](http://anastruct.readthedocs.io/en/latest/?badge=latest)

Analyse 2D Frames and trusses for slender structures. Determine the bending moments, shear forces, axial forces and displacements.

## Installation

For the actively developed version:
```
$ pip install git+https://github.com/ritchie46/anaStruct.git
```

Or for a release:
```
$ pip install anastruct
```

## Documentation

### Real world use case!
[Non linear water accumulation analysis](https://ritchievink.com/blog/2017/08/23/a-nonlinear-water-accumulation-analysis-in-python/)

### Simple examples.

[code examples!](https://ritchievink.com/blog/2017/01/12/python-1d-fem-example-1/)

### Reference guide

[reference](http://anastruct.readthedocs.io)

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
* geometrical non linearity :heavy_check_mark:

```python
from anastruct.fem.system import SystemElements

ss = SystemElements(EA=15000, EI=5000)

# Add beams to the system.
ss.add_element(location=[[0, 0], [0, 5]])
ss.add_element(location=[[0, 5], [5, 5]])
ss.add_element(location=[[5, 5], [5, 0]])

# Add a fixed support at node 1.
ss.add_support_fixed(node_id=1)

# Add a rotational spring support at node 4.
ss.add_support_spring(node_id=4, translation=3, k=4000)

# Add loads.
ss.point_load(Fx=30, node_id=2)
ss.q_load(q=-10, element_id=2)

# Solve
ss.solve()

# Get visual results.
ss.show_structure()
ss.show_reaction_force()
ss.show_axial_force()
ss.show_shear_force()
ss.show_bending_moment()
ss.show_displacement()
```
