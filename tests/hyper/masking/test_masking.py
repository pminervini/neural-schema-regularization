# -*- coding: utf-8 -*-

import numpy as np

import hyper.masking.util as util

import unittest


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.rs = np.random.RandomState(1)

    def test_masking(self):
        triples = [
            ('a', 'p', 'a'),
            ('a', 'p', 'b'),
            ('b', 'p', 'c')]

        entities = ['a', 'b', 'c']

        embedding_lengths = util.get_embedding_lengths(triples, [1, 2, 3], [4, 5, 6])
        self.assertTrue(embedding_lengths == {'a': 6, 'b': 5, 'c': 4})

        mask = util.create_mask(3, 10, [embedding_lengths[e] for e in entities])
        true_mask = np.array([
            [ 1,  1,  1],
            [ 1,  1,  1],
            [ 1,  1,  1],
            [ 1,  1,  1],
            [ 1,  1,  0],
            [ 1,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [ 0,  0,  0]
        ])

        self.assertTrue(sum(sum(abs(mask - true_mask))) == 0)


if __name__ == '__main__':
    unittest.main()
