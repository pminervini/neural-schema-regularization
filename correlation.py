#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.signal.conv


def f(a, b):
    s = a.shape[0]
    cc = np.zeros(s)
    for i in range(s):
        _cc = .0
        for j in range(s):
            _cc += a[j] * b[(i + j) % s]
        cc[i] = _cc
    return cc


def circular_cross_correlation(x, y):
    corr_expr = T.signal.conv.conv2d(x.reshape([1, -1]), y[::-1].reshape([1, -1]),
                                     border_mode='full')[0, :]
    diff = corr_expr.shape[-1] - x.shape[0]
    beginning = corr_expr[:diff + 1]
    ending = T.concatenate([corr_expr[diff + 1:], T.zeros(1)])
    ans = (beginning + ending)[::-1]
    return ans


x = T.dvector()
y = T.dvector()
cc = theano.function([x, y], circular_cross_correlation(x, y))

a, b = np.array([63, 23, 12, 27]), np.array([84, 24, 66, 32])
print(cc(a, b))

print(f(a, b))

X = np.array([[1, 2, 3], [4, 5, 6]])
#print(X.shape)
