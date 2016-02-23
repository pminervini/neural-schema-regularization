# -*- coding: utf-8 -*-

import numpy as np

import theano.tensor as T
import theano.tensor.signal.conv


def circular_cross_correlation_numpy(a, b):
    return np.array([sum(a[j] * b[(i + j) % a.shape[0]] for j in range(a.shape[0])) for i in range(a.shape[0])])


def circular_cross_correlation_theano(x, y):
    corr_expr = T.signal.conv.conv2d(x.reshape([1, -1]), y[::-1].reshape([1, -1]),
                                     border_mode='full')[0, :]
    diff = corr_expr.shape[-1] - x.shape[0]
    beginning = corr_expr[:diff + 1]
    ending = T.concatenate([corr_expr[diff + 1:], T.zeros(1)])
    return (beginning + ending)[::-1]
