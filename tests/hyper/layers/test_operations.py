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
        f = theano.function([x, y], operations.circular_cross_correlation(x, y))

        a, b = np.array([63, 23, 12, 27]), np.array([84, 24, 66, 32])

        th_value = f(a, b)
        np_value = operations.circular_cross_correlation_numpy(a, b)

        self.assertTrue(len(th_value) == len(np_value))
        for th_elem, np_elem in zip(th_value, np_value):
            self.assertTrue(th_elem == np_elem)

    def test_scan(self):
        ss, os = T.matrix(), T.matrix()
        res, _ = theano.scan(lambda s, o: operations.circular_cross_correlation(s, o),
                             sequences=[ss, os])

        f = theano.function([ss, os], res)

        _ss = np.array([[63, 23, 12, 27], [63, 23, 12, 27]])
        _os = np.array([[84, 24, 66, 32], [84, 24, 66, 32]])

        _res = f(_ss, _os)

        for i in range(_ss.shape[1]):
            self.assertTrue(_res[0, i] == _res[1, i])

if __name__ == '__main__':
    unittest.main()
