Loads
=====

anaStruct allows the following loads on a structure. There are loads on nodes and loads on elements. Element loads are
implicitly placed on the loads and recalculated during post processing.

Node loads
----------

point_load
##########

Point loads are defined in x- and/ or y-direction, or by defining a load with an angle.

.. automethod:: anastruct.fem.system.SystemElements.point_load

moment_load
###########

Moment loads apply a rotational force on the nodes.

.. automethod:: anastruct.fem.system.SystemElements.moment_load

Element loads
-------------

Q-loads are distributed loads. They can act perpendicular to the elements direction, parallel to the elements direction,
and in global x and y directions.

q_load
######

.. automethod:: anastruct.fem.system.SystemElements.q_load

