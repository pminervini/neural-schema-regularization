# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.embeddings import Embedding

from hyper import regularizers
from hyper import constraints

import unittest
import sys


def zero_loss(y_true, y_pred):
    return 0. * y_true[0, 0, 0] * y_pred[0, 0, 0]


class TestRegularizers(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_translation(self):
        #return

        r = regularizers.TranslationRuleRegularizer([0], [(1, False), (2, False)], l=1.)
        #r = regularizers.TranslationRuleRegularizer([0], [(1, False)], l=1.)

        model = Sequential()
        embedding_layer = Embedding(input_dim=3, output_dim=10, input_length=3,
                                    W_regularizer=r, W_constraint=constraints.NormConstraint(1.))
        model.add(embedding_layer)

        model.compile(loss=zero_loss, optimizer='adagrad')

        X, y = np.zeros((32, 3)), np.zeros((32, 1, 1))
        model.fit(X, y, batch_size=32, nb_epoch=10000, verbose=0)

        W = embedding_layer.trainable_weights[0].get_value()

        d = np.sum(abs(W[0, :] - (W[1, :] + W[2, :])))

        #print('test_translation: ', d)

        self.assertTrue(d < 0.01)

    def test_group(self):
        #return

        reg_1 = regularizers.TranslationRuleRegularizer([0], [(1, False), (2, False)], l=1.)
        reg_2 = regularizers.TranslationRuleRegularizer([2], [(3, False)], l=1.)

        r = regularizers.GroupRegularizer([reg_1, reg_2])

        model = Sequential()
        embedding_layer = Embedding(input_dim=4, output_dim=10, input_length=None,
                                    W_regularizer=r, W_constraint=constraints.NormConstraint(1.))
        model.add(embedding_layer)

        model.compile(loss=zero_loss, optimizer='adagrad')

        X, y = np.zeros((32, 3)), np.zeros((32, 1, 1))
        model.fit(X, y, batch_size=32, nb_epoch=10000, verbose=0)

        W = embedding_layer.trainable_weights[0].get_value()
        d_1 = np.sum(abs(W[0, :] - (W[1, :] + W[2, :])))
        d_2 = np.sum(abs(W[2, :] - W[3, :]))

        #print('test_group: ', d_1, d_2)

        self.assertTrue(d_1 < 0.01)
        self.assertTrue(d_2 < 0.01)

    def test_stress(self):
        #return

        n = 10
        rs = [regularizers.TranslationRuleRegularizer([i], [(i + 1, False)], l=0.1) for i in range(n)]

        r = regularizers.GroupRegularizer(rs)

        model = Sequential()
        embedding_layer = Embedding(input_dim=n + 1, output_dim=10, input_length=None, init='glorot_uniform',
                                    W_regularizer=r, W_constraint=constraints.NormConstraint(1.))
        model.add(embedding_layer)

        #sys.setrecursionlimit(65536)

        model.compile(loss=zero_loss, optimizer='adagrad')

        X, y = np.zeros((32, 3)), np.zeros((32, 1, 1))
        model.fit(X, y, batch_size=32, nb_epoch=100000, verbose=0)

        W = embedding_layer.trainable_weights[0].get_value()

        #print(W)

        d = np.sum(abs(W[0, :] - W[n, :]))

        #print('test_stress: ', d)

        self.assertTrue(d < 0.01)


if __name__ == '__main__':
    unittest.main()
