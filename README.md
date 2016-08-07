## Installation

For the latest version download this repository and unzip the file. Open the command box in the unzipped file location and run:
```
$ python setup.py install
```

## 2D FEM Frames and Trusses
![](images/rand/structure.png)

### See the wiki for some examples.

* __[Read the docs!](https://github.com/ritchie46/structural_engineering/wiki)__


## Concrete M-kappa diagram

M-kappa of a concrete cross section:

```python
import StructuralEngineering.concrete as se
# Define the concrete material
g_concrete = se.MaterialConcrete(
    fck=20,  # characteristic compression stress capacity
    fctk=1.545,  # characteristic tensile stress capacity
)

# Create the stress-strain-diagram
g_concrete.det_bi_linear_diagram()

# Define the rebar material
g_rebar = se.MaterialRebar(
    fyk=500,
)

# Define the reinforced concrete cross section
# beam 300 * 500
cs = se.ReinforcedConcrete(
    coordinate_list=[[0, 0], [0, 500], [300, 500], [300, 0], [0, 0]],
    materialConcrete=g_concrete,
    materialRebar=g_rebar)

# add rebar
cs.add_rebar(
    n=2,  # number of bars
    diam=12,  # diameter of the bars
    d=400)  # distance from the top of the cross section

cs.plot_M_Kappa()
```