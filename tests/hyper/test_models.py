# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.constraints import Constraint, MaxNorm
from keras.layers.core import Merge
from keras import backend as K
import hyper.layers.core

import sys
import unittest
import logging


class FixedNorm(Constraint):
    def __init__(self, m=1., axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        unit_norm = p / (norms + 1e-7)
        return unit_norm * self.m

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}


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

            #merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
            #model.add(merge_layer)

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

            norm_constraint = FixedNorm(m=1.5, axis=1)
            entity_layer = Embedding(input_dim=3, output_dim=10, input_length=None, weights=[W_emb],
                                     W_constraint=norm_constraint)
            #print(len(entity_layer.constraints))

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

            #merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
            #model.add(merge_layer)

            merge_layer = Merge([predicate_encoder, entity_encoder], mode=f, output_shape=lambda _: (None, 1))
            model.add(merge_layer)

            model.compile(loss='binary_crossentropy', optimizer='adagrad')

            Xr = np.array([[1]])
            Xe = np.array([[1, 2]])
            y = np.array([0])

            model.fit([Xr, Xe], y, batch_size=1, nb_epoch=10, verbose=0)

            #print(entity_layer.constraints, type(entity_layer.constraints))
            normalized_embeddings = entity_layer.trainable_weights[0].get_value()

            #print(np.linalg.norm(normalized_embeddings[0, :]))

            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[0, :]) - 1.5) < 1e-6)
            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[1, :]) - 1.5) < 1e-6)
            self.assertTrue(abs(np.linalg.norm(normalized_embeddings[2, :]) - 1.5) < 1e-6)

    def test_unitnorm(self):
        rs = np.random.RandomState(1)

        for _ in range(2):
            x = rs.rand()

            example_array = np.random.random((100, 100)) * 100. - 50.
            example_array[0, 0] = 0.

            unitnorm_instance = FixedNorm(m=x, axis=0)
            normalized = unitnorm_instance(K.variable(example_array))
            norm_of_normalized = np.sqrt(np.sum(K.eval(normalized) ** 2, axis=0))

            # in the unit norm constraint, it should be equal to x
            difference = norm_of_normalized - x

            largest_difference = np.max(np.abs(difference))
            self.assertTrue(np.abs(largest_difference) < 10e-5)


if __name__ == '__main__':
    unittest.main()
