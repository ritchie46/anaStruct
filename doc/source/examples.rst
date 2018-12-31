Examples
========

Simple
#######

.. code-block:: python

    ss = SystemElements(EA=5000)
    ss.add_truss_element(location=[[0, 0], [0, 5]])
    ss.add_truss_element(location=[[0, 5], [5, 5]])
    ss.add_truss_element(location=[[5, 5], [5, 0]])
    ss.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * math.sqrt(2))

    ss.add_support_hinged(node_id=1)
    ss.add_support_hinged(node_id=4)

    ss.point_load(Fx=10, node_id=2)

    ss.solve()
    ss.show_structure()
    ss.show_reaction_force()
    ss.show_axial_force()
    ss.show_displacement(factor=10)

.. image:: img/examples/truss_struct.png

.. image:: img/examples/truss_react.png

.. image:: img/examples/truss_axial.png

.. image:: img/examples/truss_displa.png


Intermediate
############

.. code-block:: python

    from anastruct import SystemElements
    import numpy as np

    ss = SystemElements()
    element_type = 'truss'

    # Create 2 towers
    width = 6
    span = 30
    k = 5e3

    # create triangles
    y = np.arange(1, 10) * np.pi
    x = np.cos(y) * width * 0.5
    x -= x.min()

    for length in [0, span]:
        ss.add_element_grid(x + length, y, element_type=element_type)
        x_left_column = np.ones(y[::2].shape) * x.min() + length
        x_right_column = np.ones(y[::2].shape) * x.max() + length
        ss.add_element_grid(x_left_column, y[::2])
        ss.add_element_grid(x_right_column, y[::2])

        ss.add_support_spring(
            node_id=ss.find_node_id(vertex=[x_left_column[0], y[0]]),
            translation=2,
            k=k)
        ss.add_support_spring(
            node_id=ss.find_node_id(vertex=[x_right_column[0], y[0]]),
            translation=2,
            k=k)

    # add top girder
    ss.add_element_grid([0, width, span, span + width], np.ones(4) * y.max())

    for el in ss.element_map.values():
        if np.isclose(np.sin(el.ai), 1):
            ss.q_load(
                q=1,
                element_id=el.id,
                direction='x'
            )

    ss.show_structure()
    ss.solve()
    ss.show_displacement()

.. image:: img/examples/tower_bridge_struct.png

.. image:: img/examples/tower_bridge_displa.png

Advanced
#########
Take a look at this blog post. Here anaStruct was used to do a non linear water accumulation analysis.

`Water accumulation blog post <https://www.ritchievink.com/blog/2017/08/23/a-nonlinear-water-accumulation-analysis-in-python/>`_.