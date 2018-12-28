Supports
========

The following kinds of support conditions are possible.

* hinged (the node is able to rotate, but cannot translate)
* roll (the node is able to rotate and translation is allowed in one direction)
* fixed (the node cannot translate and not rotate)
* spring (translation and rotation are allowed but only with a linearly increasing resistance)

add_support_hinged
##################

.. automethod:: anastruct.fem.system.SystemElements.add_support_hinged

add_support_roll
################

.. automethod:: anastruct.fem.system.SystemElements.add_support_roll

add_support_fixed
#################

.. automethod:: anastruct.fem.system.SystemElements.add_support_fixed

add_support_spring
##################

.. automethod:: anastruct.fem.system.SystemElements.add_support_spring