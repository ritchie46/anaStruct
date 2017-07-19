import numpy as np


def find_closest_index(array, searchFor):
    """
    Find the closest index in an ordered array
    :param array: array searched in
    :param searchFor: value that is searched
    :return: index of the closest match
    """
    if len(array) == 1:
        return 0

    # determine the direction of the iteration
    if array[1] < array[0] > searchFor:  # iteration from big to small

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return 0
        elif array[0] > searchFor and array[-1] > searchFor:
            return len(array) - 1

        for i in range(len(array)):
            if array[i] <= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                # search for the index with the smallest absolute difference
                if searchFor - valueAtIndex <= previousValue - searchFor:
                    return i
                else:
                    return i - 1

    else:
        # array[0] is smaller than searchFor
        # iteration from small to big

        # check if the value does not go out of scope
        if array[0] < searchFor and array[-1] < searchFor:
            return len(array) - 1
        elif array[0] > searchFor and array[-1] > searchFor:
            return 0

        for i in range(len(array)):
            if array[i] >= searchFor:
                valueAtIndex = array[i]
                previousValue = array[i - 1]

                if valueAtIndex - searchFor <= searchFor - previousValue:
                    return i
                else:
                    return i - 1


def is_moving_towards(test_node, node_position, displacement):
    """
    Tests if a node is displacing towards or away from the node.
    :param test_node: Nodes that stands still (vector)
    :param node_position: Node of which the displacment tested (vector)
    :param displacement: (vector)
    :return: (Boolean)
    """
    to_node = test_node - node_position
    return np.dot(to_node, displacement) > 0


def find_nearest(array, value):
    """
    :param array: (numpy array object)
    :param value: (float) value searched for
    :return: (tuple) nearest value, index
    """
    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index = (np.abs(array-value)).argmin()
    return array[index], index


def converge(lhs, rhs, div=3):
    """
    Determine convergence factor.

    :param lhs: (flt)
    :param rhs: (flt)
    :param div: (flt)
    :return: multiplication factor (flt) ((lhs / rhs) - 1) / div + 1
    """
    return (np.abs(rhs) / np.abs(lhs) - 1) / div + 1
