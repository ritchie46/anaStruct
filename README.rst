Structural Engineering
======================

*Will be a small library with solutions for Structural Engineering problems*

Installation
============

::

    $ pip install structural_engineering

For now you're able to create a cross section of any polygon and determine:
 - Moment of Inertia
 - Center of Gravity

Concrete calculation
 - Concrete cross section calculation for multiple layers of rebar
 - M-Kappa (concrete)

Coming up:
 - LE (Resisting Moment) any material.
 - FRP reinforcement

Create a polygon cross section:

.. code:: python

	import cross_section.py
	# create the object, the parameters is a list containing the the points of the polygon in [x, z]
	cs = CrossSection([[0, 0], [0, 500], [300, 500], [250, 250], [500, 0], [0, 0]])
	print(cs.moment_of_inertia)

	# visual of the result
	cs.print_in_lines()


	M-kappa of a concrete cross section:
	.. code:: python
	# Define the concrete material
	g_concrete = MaterialConcrete(
	fck=20,  # characteristic compression stress capacity
	fctk=1.545,  # characteristic tensile stress capacity
	)

	# Create the stress-strain-diagram
	g_concrete.det_bi_linear_diagram()

	# Define the rebar material
	g_rebar = MaterialRebar(
	fyk=500,
	)

	# Define the reinforced concrete cross section
	# beam 300 * 500
	cs = ReinforcedConcrete(
	coordinate_list=[[0, 0], [0, 500], [300, 500], [300, 0], [0, 0]],
	materialConcrete=g_concrete,
	materialRebar=g_rebar)

	# add rebar
	cs.add_rebar(
	n=2,  # number of bars
	diam=12,  # diameter of the bars
	d=400)  # distance from the top of the cross section

	cs.plot_M_Kappa()
