# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, LambdaMerge

import unittest


def func(X):
    relation_embedding = X[0]
    entity_embeddings = X[1]
    return relation_embedding + entity_embeddings[:, 0, 0:2]


class TestGraph(unittest.TestCase):
    def setUp(self):
        pass

    def test_graph(self):
        graph = Graph()

        graph.add_input(name='input_entities', input_shape=(None,), dtype='int32')
        graph.add_input(name='input_relation', input_shape=(1,), dtype='int32')

        W = np.ones((5, 2))
        entity_embedding_layer = Embedding(input_dim=5, output_dim=2, weights=[W], input_length=None)
        graph.add_node(entity_embedding_layer, name='entity_embeddings', input='input_entities')

        W = np.ones((3, 2))
        relation_embedding_layer = Embedding(input_dim=3, output_dim=2, weights=[W], input_length=1)
        graph.add_node(relation_embedding_layer, name='relation_embedding', input='input_relation')

        #merge_layer = LambdaMerge([relation_embedding_layer, entity_embedding_layer], function=func)
        #graph.add_node(merge_layer, name='merge_layer')

        #graph.add_output(name='output', input='merge_layer')

        #graph.compile(optimizer='rmsprop', loss={'output': 'categorical_crossentropy'})

if __name__ == '__main__':
    unittest.main()