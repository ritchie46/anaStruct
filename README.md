# Structural Engineering

*Will be a small library with solutions for Structural Engineering problems*

For now your able to create cross section of any kind and determine:
* Moment of Inertia
* Center of Gravity

Coming up:
* Concrete cross section calculation for multiple layers of rebar and FRP (Resisting Moment).
* LE (Resisting Moment) any material.

Required:
* numpy and matplotlib

Create a random cross section:
````
import cross_section.py
# create the object, the parameters is a list containing the the points of the polygon in [x, z]
a = CrossSection([[0, 0], [0, 500], [300, 500], [250, 250], [500, 0], [0, 0]])
print(a.moment_of_inertia)

# visual of the result
a.print_in_lines()