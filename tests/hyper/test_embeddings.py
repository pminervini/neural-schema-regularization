# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras import backend as K

from hyper import constraints
from hyper.layers.embeddings import MemoryEfficientEmbedding, Frame

import unittest


def loss(y_true, y_pred):
    loss_value = K.sum(y_true) + K.sum(y_pred)
    return loss_value


class TestEmbeddings(unittest.TestCase):
    _ITERATIONS = 8

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

            frames = [Frame(0, 1, 0, 2), Frame(2, 3, 2, 4)]
            weights = [self.rs.random_sample((1, 2)), np.ones((1, 2))]

            layer = MemoryEfficientEmbedding(input_dim=3, output_dim=10, input_length=None,
                                             frames=frames, weights=weights)
            encoder.add(layer)
            encoder.compile(loss=loss, optimizer='adagrad')

            x = [np.array([[0, 1, 2]])]

            old_y = encoder.predict(x, batch_size=1)[0]
            old_weights = encoder.get_weights()
            #print(old_y)

            for frame, weight in zip(frames, weights):
                for i in range(frame.row_end - frame.row_start):
                    for j in range(frame.col_end - frame.col_start):
                        self.assertAlmostEquals(weight[i, j], old_y[frame.row_start + i, frame.col_start + j])

            t = [np.zeros(shape=(1, 3, 10))]
            encoder.fit(x=x, y=t, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)

            new_y = encoder.predict(x, batch_size=1)[0]
            new_weights = encoder.get_weights()

            for old_weight, new_weight in zip(old_weights, new_weights):
                self.assertTrue(sum(sum(old_weight - new_weight)) > .0)

    def test_memory_efficient_embeddings_constraint(self):
        for _ in range(self._ITERATIONS):
            encoder = Sequential()

            frames = [Frame(0, 1, 0, 2), Frame(2, 3, 2, 4)]
            weights = [self.rs.random_sample((1, 2)), np.ones((1, 2))]

            normcon = constraints.NormConstraint(m=1., axis=1)

            layer = MemoryEfficientEmbedding(input_dim=3, output_dim=10, input_length=None,
                                             frames=frames, weights=weights, W_constraint=normcon)
            encoder.add(layer)

            optimizer = SGD(lr=.0, momentum=.0, decay=.0, nesterov=False)
            encoder.compile(loss=loss, optimizer=optimizer)

            x = [np.array([[0, 1, 2]])]

            old_y = encoder.predict(x, batch_size=1)[0]
            old_weights = encoder.get_weights()

            t = [np.zeros(shape=(1, 3, 10))]
            encoder.fit(x=x, y=t, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)

            new_y = encoder.predict(x, batch_size=1)[0]
            new_weights = encoder.get_weights()

            for weight in new_weights:
                self.assertAlmostEquals(np.linalg.norm(weight, 2), 1.0, 5)


if __name__ == '__main__':
    unittest.main()
