# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge

import hyper.layers.core
from hyper.constraints import norm

import sys
import unittest


class TestModels(unittest.TestCase):

    def setUp(self):
        pass

    def test_translating_embeddings_inference(self):
        rs = np.random.RandomState(1)

        for _ in range(32):
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

            merge_layer = Merge([predicate_encoder, entity_encoder], mode=f, output_shape=lambda _: (None, 1))
            model.add(merge_layer)

            model.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xr = np.array([[1]])
            Xe = np.array([[1, 2]])

            y = model.predict([Xr, Xe], batch_size=1)

            expected = - np.sum(np.abs(W_emb[1, :] + W_pred[1, :] - W_emb[2, :]))
            self.assertTrue(abs(y[0, 0] - expected) < 1e-6)

    def test_translating_embeddings_learning(self):
        rs = np.random.RandomState(1)

        for _ in range(2):
            W_pred, W_emb = rs.random_sample((3, 10)), rs.random_sample((3, 10))

            predicate_encoder = Sequential()
            entity_encoder = Sequential()

            predicate_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_pred])
            predicate_encoder.add(predicate_layer)

            norm_constraint = norm(m=1.5, axis=1)
            entity_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_emb],
                                     W_constraint=norm_constraint)

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

            merge_layer = Merge([predicate_encoder, entity_encoder], mode=f, output_shape=lambda _: (None, 1))
            model.add(merge_layer)

            model.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xr = np.array([[1]])
            Xe = np.array([[1, 2]])
            y = np.array([0])

            model.fit([Xr, Xe], y, batch_size=1, nb_epoch=10, verbose=0)

            normalized_embeddings = entity_layer.trainable_weights[0].get_value()

            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[0, :]) - 1.5) < 1e-6)
            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[1, :]) - 1.5) < 1e-6)
            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[2, :]) - 1.5) < 1e-6)


if __name__ == '__main__':
    unittest.main()
