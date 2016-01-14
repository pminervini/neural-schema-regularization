# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from hyper import similarities

import math
import unittest


def sim(a, b, f):
    return f(a, b, axis=1)


class TestSimilarities(unittest.TestCase):
    nb_samples = 100
    input_dim = 32

    def setUp(self):
        pass

    def test_similarities(self):
        for sim_f in [similarities.L1, similarities.L2, similarities.L2SQR, similarities.DOT, similarities.COSINE]:
            for _ in range(32):
                x = np.random.randn(self.nb_samples, self.input_dim).astype(K.floatx())
                y = np.random.randn(self.nb_samples, self.input_dim).astype(K.floatx())

                X = K.placeholder(ndim=2)
                Y = K.placeholder(ndim=2)

                function = K.function([X, Y], [sim(X, Y, f=sim_f)])

                s1, s2 = function([x, x])[0], function([y, y])[0]
                d1, d2 = function([x, y])[0], function([y, x])[0]

                for s in [s1, s2]:
                    for d in [d1, d2]:
                        assert((False in (s > d)) is False)

    def test_l1(self):
        x_values = np.random.rand(1024)
        y_values = np.random.rand(1024)

        sim_f = similarities.L1

        X, Y = K.placeholder(ndim=2), K.placeholder(ndim=2)
        function = K.function([X, Y], [sim(X, Y, f=sim_f)])

        for x, y in zip(x_values, y_values):
            assert(abs(function([[[x]], [[y]]])[0] - (- abs(x - y))) < 1e-6)

    def test_l2(self):
        x_values = np.random.rand(1024)
        y_values = np.random.rand(1024)

        sim_f = similarities.L2

        X, Y = K.placeholder(ndim=2), K.placeholder(ndim=2)
        function = K.function([X, Y], [sim(X, Y, f=sim_f)])

        for x, y in zip(x_values, y_values):
            assert(abs(function([[[x]], [[y]]])[0] - (- math.sqrt((x - y) ** 2))) < 1e-6)

    def test_l2sqr(self):
        x_values = np.random.rand(1024)
        y_values = np.random.rand(1024)

        sim_f = similarities.L2SQR

        X, Y = K.placeholder(ndim=2), K.placeholder(ndim=2)
        function = K.function([X, Y], [sim(X, Y, f=sim_f)])

        for x, y in zip(x_values, y_values):
            assert(abs(function([[[x]], [[y]]])[0] - (- (x - y) ** 2)) < 1e-6)

    def test_dot(self):
        x_values = np.random.rand(1024)
        y_values = np.random.rand(1024)

        sim_f = similarities.DOT

        X, Y = K.placeholder(ndim=2), K.placeholder(ndim=2)
        function = K.function([X, Y], [sim(X, Y, f=sim_f)])

        for x, y in zip(x_values, y_values):
            assert(abs(function([[[x]], [[y]]])[0] - (x * y)) < 1e-6)

    def test_cosine(self):
        x_values = np.random.rand(1024)
        y_values = np.random.rand(1024)

        sim_f = similarities.COSINE

        X, Y = K.placeholder(ndim=2), K.placeholder(ndim=2)
        function = K.function([X, Y], [sim(X, Y, f=sim_f)])

        for x, y in zip(x_values, y_values):
            cos = x * y / (math.sqrt(x ** 2) * math.sqrt(y ** 2))
            assert(abs(function([[[x]], [[y]]])[0] - cos) < 1e-6)


if __name__ == '__main__':
    unittest.main()
