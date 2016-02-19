# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding

from hyper import regularizers
from hyper import constraints

import unittest


class TestRegularizers(unittest.TestCase):

    def setUp(self):
        pass

    def test_translation_rule_regularizer(self):
        r = regularizers.TranslationRuleRegularizer([0], [(1, False), (2, False)], l=1.)

        model = Sequential()
        embedding_layer = Embedding(input_dim=3, output_dim=10, input_length=None,
                                    W_regularizer=r, W_constraint=constraints.NormConstraint(1.))
        model.add(embedding_layer)

        def zero_loss(y_true, y_pred):
            return .0 * (y_true.sum() + y_pred.sum())

        model.compile(loss=zero_loss, optimizer='adagrad')

        X, y = np.zeros((1024, 3)), np.zeros((1024, 1, 1))
        model.fit(X, y, batch_size=1024, nb_epoch=10000, verbose=0)

        W = embedding_layer.trainable_weights[0].get_value()

        d = np.sum(abs(W[0, :] - (W[1, :] + W[2, :])))

        print(d)

        self.assertTrue(d < 0.01)

    def test_group_regularizer(self):
        reg_1 = regularizers.TranslationRuleRegularizer([0], [(1, False), (2, False)], l=1.)
        reg_2 = regularizers.TranslationRuleRegularizer([2], [(3, False)], l=1.)

        r = regularizers.GroupRegularizer([reg_1, reg_2])

        model = Sequential()
        embedding_layer = Embedding(input_dim=4, output_dim=10, input_length=None,
                                    W_regularizer=r, W_constraint=constraints.NormConstraint(1.))
        model.add(embedding_layer)

        def zero_loss(y_true, y_pred):
            return .0 * (y_true.sum() + y_pred.sum())

        model.compile(loss=zero_loss, optimizer='adagrad')

        X, y = np.zeros((32, 3)), np.zeros((32, 1, 1))
        model.fit(X, y, batch_size=32, nb_epoch=10000, verbose=0)

        W = embedding_layer.trainable_weights[0].get_value()
        d_1 = np.sum(abs(W[0, :] - (W[1, :] + W[2, :])))
        d_2 = np.sum(abs(W[2, :] - W[3, :]))

        self.assertTrue(d_1 < 0.01)
        self.assertTrue(d_2 < 0.01)


if __name__ == '__main__':
    unittest.main()
