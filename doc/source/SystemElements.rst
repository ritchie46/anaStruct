SystemElements
==============

.. autoclass:: anastruct.fem.system.SystemElements


Add elements
-------------


    .. automethod:: anastruct.fem.system.SystemElements.add_truss_element

    .. automethod:: anastruct.fem.system.SystemElements.add_element

    .. automethod:: anastruct.fem.system.SystemElements.add_multiple_elements


Apply forces
------------
    
    .. automethod:: anastruct.fem.system.SystemElements.point_load

    .. automethod:: anastruct.fem.system.SystemElements.q_load

    .. automethod:: anastruct.fem.system.SystemElements.moment_load


Supporting conditions
---------------------

    .. automethod:: anastruct.fem.system.SystemElements.add_support_hinged

    .. automethod:: anastruct.fem.system.SystemElements.add_support_roll

    .. automethod:: anastruct.fem.system.SystemElements.add_support_fixed

    .. automethod:: anastruct.fem.system.SystemElements.add_support_spring


Visual feedback
----------------

    .. automethod:: anastruct.fem.system.SystemElements.show_structure

    .. automethod:: anastruct.fem.system.SystemElements.show_bending_moment

    .. automethod:: anastruct.fem.system.SystemElements.show_axial_force

    .. automethod:: anastruct.fem.system.SystemElements.show_shear_force

    .. automethod:: anastruct.fem.system.SystemElements.show_reaction_force

    .. automethod:: anastruct.fem.system.SystemElements.show_displacement


Numerical feedback
------------------

    .. automethod:: anastruct.fem.system.SystemElements.get_node_results_system

    .. automethod:: anastruct.fem.system.SystemElements.get_node_displacements

    .. automethod:: anastruct.fem.system.SystemElements.get_element_results

    .. automethod:: anastruct.fem.system.SystemElements.get_element_result_range

    .. automethod:: anastruct.fem.system.SystemElements.get_element_result_range

    .. automethod:: anastruct.fem.system.SystemElements.find_node_id

    .. automethod:: anastruct.fem.system.SystemElements.nodes_range
