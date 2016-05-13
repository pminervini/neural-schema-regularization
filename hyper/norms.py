# -*- coding: utf-8 -*-

from keras import backend as K
import sys


def l1_norm(x, axis=1):
    """
    L1 Norm.

    .. math:: \\sum_i \\abs(x_i)

    :param x: First term.
    :param axis: Axis.
    :return: Norm Value.
    """
    return K.sum(K.abs(x), axis=axis)


def l2_norm(x, axis=1):
    """
    L2 Norm.

    .. math:: \\sqrt{\\sum_i (x_i)^2}

    :param x: First term.
    :param axis: Axis.
    :return: Norm Value.
    """
    return K.sqrt(K.sum(K.square(x), axis=axis))


def square_l2_norm(x, axis=1):
    """
    L2 Norm.

    .. math:: \\sum_i (x_i)^2

    :param x: First term.
    :param axis: Axis.
    :return: Norm Value.
    """
    return K.sum(K.square(x), axis=axis)


# aliases
l1 = L1 = l1_norm
l2 = L2 = l2_norm
square_l2 = SQR_L2 = square_l2_norm


def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown norm: %s" % function_name)
    return function
