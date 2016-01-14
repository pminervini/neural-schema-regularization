# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda

import unittest


def sum_f(X, axis=1):
    return K.sum(X, axis=axis)


def prod_f(X, axis=1):
    return K.prod(X, axis=axis)


class TestLambda(unittest.TestCase):
    def setUp(self):
        pass

    def test_sum(self):
        model = Sequential()

        W = np.ones((3, 2))

        model.add(Embedding(input_dim=3, output_dim=2, weights=[W]))
        model.add(Lambda(sum_f))

        model.compile(loss='binary_crossentropy', optimizer='sgd')

        X = np.asarray([[0, 1, 0, 2]])
        Y = model.predict(X, batch_size=1)[0]

        self.assertTrue(abs(Y[0] - 4.) < 1e-8)
        self.assertTrue(abs(Y[1] - 4.) < 1e-8)

    def test_prod(self):
        model = Sequential()

        W = np.ones((3, 2))

        model.add(Embedding(input_dim=3, output_dim=2, weights=[W]))
        model.add(Lambda(prod_f))

        model.compile(loss='binary_crossentropy', optimizer='sgd')

        X = np.asarray([[0, 1, 0, 2]])
        Y = model.predict(X, batch_size=1)[0]

        self.assertTrue(abs(Y[0] - 1.) < 1e-8)
        self.assertTrue(abs(Y[1] - 1.) < 1e-8)

if __name__ == '__main__':
    unittest.main()
