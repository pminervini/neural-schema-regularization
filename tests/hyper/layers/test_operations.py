# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T

from hyper.layers import operations

import unittest


class TestOperations(unittest.TestCase):

    def setUp(self):
        pass

    def test_cross_correlation(self):
        x, y = T.vector(), T.vector()
        y = T.dvector()
        f = theano.function([x, y], operations.circular_cross_correlation_theano(x, y))

        a, b = np.array([63, 23, 12, 27]), np.array([84, 24, 66, 32])

        th_value = f(a, b)
        np_value = operations.circular_cross_correlation_numpy(a, b)

        assert len(th_value) == len(np_value)
        for th_elem, np_elem in zip(th_value, np_value):
            assert th_elem == np_elem

if __name__ == '__main__':
    unittest.main()
