# -*- coding: utf-8 -*-

from keras.constraints import Constraint
from keras import backend as K

from keras.utils.generic_utils import get_from_module

import logging


class NormConstraint(Constraint):
    def __init__(self, m=1., axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        unit_norm = p / (norms + 1e-7)
        return unit_norm * self.m

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}


class MaskConstraint(Constraint):
    def __init__(self, mask=None):
        self.mask = mask

    def __call__(self, p):
        return p * self.mask


norm = Norm = NormConstraint
mask = Mask = MaskConstraint


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint', instantiate=True, kwargs=kwargs)
