# -*- coding: utf-8 -*-

from keras.constraints import Constraint
from keras import backend as K


class NormConstraint(Constraint):
    def __init__(self, norm=1.):
        self.norm = norm

    def __call__(self, p):
        p_t = K.transpose(p)
        unit_norm = K.transpose(p_t / (K.sqrt(K.sum(K.square(p_t), axis=0)) + 1e-7))
        return unit_norm * self.norm

    def get_config(self):
        return {'name': self.__class__.__name__, 'norm': self.norm}
