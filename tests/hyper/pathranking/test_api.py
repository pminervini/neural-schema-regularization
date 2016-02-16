# -*- coding: utf-8 -*-

from hyper.pathranking.api import PathRankingClient

import unittest


class TestPathRankingAPI(unittest.TestCase):
    def setUp(self):
        self.client = PathRankingClient()

    def test_friends(self):
        if self.client.is_online():
            triples = [
                ('Mark', 'friendOf', 'John'),
                ('John', 'friendOf', 'Paul'),
                ('Mark', 'friendOfAFriend', 'Paul')]

            predicates = self.client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)

            for p in predicates:
                if p.predicate == 'friendOfAFriend':
                    self.assertTrue(len(p.features) == 1)
                    feature = p.features[0]
                    self.assertTrue(len(feature.hops) == 2)
                    hops = feature.hops
                    for hop in hops:
                        self.assertTrue(hop.predicate == 'friendOf' and hop.is_inverse is False)
                    self.assertTrue(p.weights[0] > .0)


if __name__ == '__main__':
    unittest.main()
