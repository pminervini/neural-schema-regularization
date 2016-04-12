# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge

import unittest


def func(X):
    relation_embedding = X[0]
    entity_embeddings = X[1]
    return relation_embedding + entity_embeddings[:, 0, 0:2]


class TestSequential(unittest.TestCase):
    def setUp(self):
        pass

    def test_sequential(self):
        relation_encoder = Sequential()
        entity_encoder = Sequential()

        W_rel = np.ones((3, 2)) * 2
        relation_embedding_layer = Embedding(input_dim=3, output_dim=2, weights=[W_rel], input_length=1)
        relation_encoder.add(relation_embedding_layer)

        W_ent = np.ones((5, 3))
        entity_embedding_layer = Embedding(input_dim=5, output_dim=3, weights=[W_ent], input_length=None)
        entity_encoder.add(entity_embedding_layer)

        model = Sequential()
        merge_layer = Merge([relation_encoder, entity_encoder], mode=func, output_shape=lambda _: (None, 1))
        model.add(merge_layer)

        model.compile(loss='mse', optimizer='rmsprop')

        Xr = np.asarray([[0]])
        Xe = np.asarray([[0, 1, 0, 2]])

        Y = model.predict([Xr, Xe])

        self.assertTrue(abs(Y[0, 0, 0] - 3.) < 1e-6)

if __name__ == '__main__':
    unittest.main()