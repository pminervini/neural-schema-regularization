# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding

import unittest


class TestLookUp(unittest.TestCase):
    def setUp(self):
        pass

    def test_lookup(self):
        lookup = Sequential()
        lookup.add(Embedding(3, 2))

        lookup.compile(loss='binary_crossentropy', optimizer='sgd')

        X = np.asarray([[0, 1, 0, 2, 1]])
        Y = lookup.predict(X, batch_size=1)[0]

        assert(Y.shape[0] == 5 and Y.shape[1] == 2)

        assert((Y[0, :] - Y[2, :]).mean() < 1e-8)
        assert((Y[1, :] - Y[4, :]).mean() < 1e-8)

if __name__ == '__main__':
    unittest.main()
