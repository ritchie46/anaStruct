Structural Engineering
======================

*A small library with solutions for Structural Engineering problems.*

Installation
============

::

    $ pip install StructuralEngineering


FEM method for 2D frames (Matrix Frames) (Work in Progress)
===========================================================
 - Bending moment (visual plot), works
 - Normal force (visual plot), works
 - Shear force (visual plot), coming up
 - Numeral results, coming up
 
 .. code:: python

    import StructuralEngineering.FEM.system as se

    # create a new system
    system = se.SystemElements()

    # add beams to the system. positive z-axis is down, positive x-axis is the right
    system.add_element(location_list=[[0, 0], [0, -5]], EA=5e3, EI=5000)
    system.add_element(location_list=[[0, -5], [5, -5]], EA=5e3, EI=5000)
    system.add_element(location_list=[[5, -5], [5, 0]], EA=5e3, EI=5000)

    # add loads to the elements and nodes
    system.q_load(elementID=2, q=10, direction=1)
    system.point_load(Fx=30, nodeID=2)

    # add supports at the nodes
    system.add_support_fixed(nodeID=1)
    system.add_support_fixed(nodeID=4)

    # solve the equations
    system.assemble_system_matrix()
    system.process_conditions()
    system.solve()

    # show the bending moment
    system.show_bending_moment()

    # show the normal force
    system.show_normal_force()

    # show the shear force
    system.show_shear_force()

Cross Sections
==============
For now you're able to create a cross section of any polygon and determine:
 - Moment of Inertia
 - Center of Gravity

 Concrete calculation
 - Concrete cross section calculation for multiple layers of rebar
 - M-Kappa (concrete)


Create a polygon cross section:

.. code:: python

    import StructuralEngineering.cross_section as cs

    # create the object, the parameters is a list containing the the points of the polygon in [x, z]
    crossSection = cs.CrossSection([[0, 0], [0, 500], [300, 500], [250, 250], [500, 0], [0, 0]])
    print(crossSection.moment_of_inertia)

    # visual of the result
    crossSection.print_in_lines()



M-kappa of a concrete cross section:

.. code:: python

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
