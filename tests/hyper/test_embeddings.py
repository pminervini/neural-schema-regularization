# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding

from hyper.layers.embeddings import MemoryEfficientEmbedding, Frame

import unittest


class TestEmbeddings(unittest.TestCase):
    _ITERATIONS = 1

    def setUp(self):
        self.rs = np.random.RandomState(1)

    def test_embeddings(self):
        for _ in range(self._ITERATIONS):
            encoder = Sequential()

            W_emb = self.rs.random_sample((3, 10))

            layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_emb])
            encoder.add(layer)

            encoder.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xe = np.array([[1, 2]])
            y = encoder.predict([Xe], batch_size=1)[0]

            self.assertTrue(sum(sum(abs(y - W_emb[[1, 2], :]))) < 1e-8)

    def test_memory_efficient_embeddings(self):
        for _ in range(self._ITERATIONS):
            encoder = Sequential()

            W_frame = self.rs.random_sample((1, 2))
            frames = [Frame(0, 1, 0, 2, W=W_frame)]

            layer = MemoryEfficientEmbedding(input_dim=3, output_dim=10, input_length=None, frames=frames)
            encoder.add(layer)

            encoder.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xe = np.array([[0, 1, 2]])
            y = encoder.predict([Xe], batch_size=1)[0]

            print(y)

if __name__ == '__main__':
    unittest.main()
