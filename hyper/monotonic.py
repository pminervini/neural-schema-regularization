# -*- coding: utf-8 -*-

import abc

from keras import backend as K


"""
Monotonically increasing and concave ranking loss functions.
"""


class Monotonic(abc.ABCMeta):
    @abc.abstractmethod
    def __call__(cls, x):
        while False:
            yield None

    @abc.abstractmethod
    def gradient(self, x):
        while False:
            yield None


class Identity(Monotonic):
    def __call__(cls, x):
        return x

    def gradient(self, x):
        return 1


class Logarithm(Monotonic):
    def __call__(cls, x):
        return K.log(1 + x)

    def gradient(self, x):
        return 2 / (x + 1)
