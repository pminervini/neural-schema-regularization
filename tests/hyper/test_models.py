# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import LambdaMerge

import hyper.layers.core

import sys
import unittest
import logging


class TestModels(unittest.TestCase):

    def setUp(self):
        pass

    def test_translating_embeddings_inference(self):
        rs = np.random.RandomState(1)

        for _ in range(64):
            W_pred, W_emb = rs.random_sample((3, 10)), rs.random_sample((3, 10))

            predicate_encoder = Sequential()
            entity_encoder = Sequential()

            predicate_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_pred])
            predicate_encoder.add(predicate_layer)

            entity_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_emb])
            entity_encoder.add(entity_layer)

            model = Sequential()

            core = sys.modules['hyper.layers.core']
            setattr(core, 'similarity function', 'L1')
            setattr(core, 'merge function', 'TransE')

            def f(args):
                import sys
                import hyper.similarities as similarities
                import hyper.layers.binary.merge_functions as merge_functions

                f_core = sys.modules['hyper.layers.core']
                similarity_function_name = getattr(f_core, 'similarity function')
                merge_function_name = getattr(f_core, 'merge function')

                similarity_function = similarities.get_function(similarity_function_name)
                merge_function = merge_functions.get_function(merge_function_name)

                return merge_function(args, similarity=similarity_function)

            merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
            model.add(merge_layer)

            model.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xr = np.array([[1]])
            Xe = np.array([[1, 2]])

            y = model.predict([Xr, Xe], batch_size=1)

            expected = - np.sum(np.abs(W_emb[1, :] + W_pred[1, :] - W_emb[2, :]))
            self.assertTrue(abs(y[0, 0] - expected) < 1e-6)

    def test_translating_embeddings_learning(self):
        rs = np.random.RandomState(1)

        for _ in range(64):
            W_pred, W_emb = rs.random_sample((3, 10)), rs.random_sample((3, 10))

            predicate_encoder = Sequential()
            entity_encoder = Sequential()

            predicate_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_pred])
            predicate_encoder.add(predicate_layer)

            entity_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_emb])
            entity_encoder.add(entity_layer)

            model = Sequential()

            core = sys.modules['hyper.layers.core']
            setattr(core, 'similarity function', 'L1')
            setattr(core, 'merge function', 'TransE')

            def f(args):
                import sys
                import hyper.similarities as similarities
                import hyper.layers.binary.merge_functions as merge_functions

                f_core = sys.modules['hyper.layers.core']
                similarity_function_name = getattr(f_core, 'similarity function')
                merge_function_name = getattr(f_core, 'merge function')

                similarity_function = similarities.get_function(similarity_function_name)
                merge_function = merge_functions.get_function(merge_function_name)

                return merge_function(args, similarity=similarity_function)

            merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
            model.add(merge_layer)

            model.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xr = np.array([[1]])
            Xe = np.array([[1, 2]])

            y = model.predict([Xr, Xe], batch_size=1)

            expected = - np.sum(np.abs(W_emb[1, :] + W_pred[1, :] - W_emb[2, :]))
            self.assertTrue(abs(y[0, 0] - expected) < 1e-6)


if __name__ == '__main__':
    unittest.main()
