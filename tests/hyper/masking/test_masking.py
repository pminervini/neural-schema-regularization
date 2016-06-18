# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

import hyper.masking.util as util

import unittest


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.rs = np.random.RandomState(1)

    def test_masking(self):
        triples = [
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 2)]

        entities = [0, 1, 2]
        embedding_lengths = util.get_embedding_lengths(triples, [1, 2, 3], [4, 5, 6])
        self.assertTrue(embedding_lengths == {0: 6, 1: 5, 2: 4})

        mask = util.create_mask(3, 10, [embedding_lengths[e] for e in entities])
        true_mask = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ])
        self.assertTrue(sum(sum(abs(mask - true_mask))) == 0)

    def test_masking_large(self):
        NE, N = 16, 10000

        entities = list(range(NE))
        triples = [(entities[self.rs.randint(NE)], 0, entities[self.rs.randint(NE)]) for _ in range(N)]

        entity_seq = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]
        entity_counts = sorted(Counter(entity_seq).items(), key=lambda entry: entry[1])

        cut_points = [1200, 1300]
        embedding_lengths = [1, 2, 3]

        entity_lengths = {}
        for (entity, count) in entity_counts:
            idx = next((cut_points.index(c) for c in cut_points if c > count), len(cut_points))
            entity_lengths[entity] = embedding_lengths[idx]

        l = [(1, 1164), (13, 1190), (4, 1219), (11, 1224), (12, 1237), (2, 1241), (8, 1241), (9, 1254), (15, 1255),
             (3, 1256), (0, 1273), (6, 1275), (10, 1275), (7, 1279), (14, 1289), (5, 1328)]
        self.assertTrue(l == entity_counts)

        l = {0: 2, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 1, 14: 2, 15: 2}
        self.assertTrue(l == entity_lengths)

        l = [2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2]
        for a, b in zip(l, [entity_lengths[i] for i in range(NE)]):
            self.assertTrue(a == b)


if __name__ == '__main__':
    unittest.main()
