# -*- coding: utf-8 -*-

import numpy as np

import theano.tensor as T
import theano.tensor.signal.conv


def circular_cross_correlation_numpy(x, y):
    return np.array([sum(x[j] * y[(i + j) % x.shape[0]] for j in range(x.shape[0])) for i in range(x.shape[0])])


def circular_cross_correlation_theano_signal(x, y):
    corr_expr = T.signal.conv.conv2d(x.reshape([1, -1]), y[::-1].reshape([1, -1]),
                                     border_mode='full')[0, :]
    diff = corr_expr.shape[-1] - x.shape[0]
    beginning = corr_expr[:diff + 1][::-1]
    ending = corr_expr[diff + 1:]
    ans = T.inc_subtensor(beginning[1:], ending[::-1])
    return ans


def circular_cross_correlation_theano_nnet(x, y):
    corr_expr = T.nnet.conv2d(x.reshape([1, 1, 1, -1]),
                              y.reshape([1, 1, 1, -1]),
                              border_mode='full',
                              filter_flip=False)[0, 0, 0, :]
    diff = corr_expr.shape[-1] - x.shape[0]
    beginning = corr_expr[:diff + 1][::-1]
    ending = corr_expr[diff + 1:]
    ans = T.inc_subtensor(beginning[1:], ending[::-1])
    return ans


def circular_cross_correlation_theano_batch(X, Y):
    corr_expr = T.nnet.conv2d(X.reshape((X.shape[0], 1, 1, X.shape[1])),
                              Y.reshape((Y.shape[0], 1, 1, Y.shape[1])),
                              border_mode='full',
                              filter_flip=False)[:, 0, 0, :]
    diff = corr_expr.shape[-1] - X.shape[1]
    beginning = corr_expr[:, :diff + 1]
    ending = T.concatenate([corr_expr[:, diff + 1:], T.zeros(1)])
    ans = (beginning + ending)[:, ::-1]
    return ans


circular_cross_correlation = circular_cross_correlation_theano_nnet
