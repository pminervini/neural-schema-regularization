# -*- coding: utf-8 -*-

from keras import backend as K
from hyper import norms

import sys


def l1_activation(x, axis=1):
    return norms.l1(x, axis=axis)


def l2_activation(x, axis=1):
    return norms.l2(x, axis=axis)


def sum_activation(x, axis=1):
    return K.sum(x, axis=axis)


# aliases
l1 = L1 = l1_activation
l2 = L2 = l2_activation
sum = SUM = sum_activation


def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown activation function: %s" % function_name)
    return function
