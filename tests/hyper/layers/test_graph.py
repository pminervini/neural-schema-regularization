# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda

import unittest


class TestGraph(unittest.TestCase):
    def setUp(self):
        pass

    def test_graph(self):
        graph = Graph()

        graph.add_input(name='input_entities', input_shape=(None,), dtype='int32')
        graph.add_input(name='input_relation', input_shape=(1,), dtype='int32')

        W = np.ones((5, 2))
        entity_embedding_layer = Embedding(input_dim=5, output_dim=2, weights=[W], input_length=None)
        graph.add_node(entity_embedding_layer, name="entity_embeddings", input="inout_entites")

        W = np.ones((3, 2))
        relation_embedding_layer = Embedding(input_dim=3, output_dim=2, weights=[W], input_length=1)
        graph.add_node(relation_embedding_layer, name="relation_embedding", input="inout_relation")





if __name__ == '__main__':
    unittest.main()