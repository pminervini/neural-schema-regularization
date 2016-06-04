# -*- coding: utf-8 -*-

from hyper.pathranking.api import PathRankingClient

import unittest

import logging


class TestPathRankingAPI(unittest.TestCase):

    def setUp(self):
        pass

    def test_friendship(self):
        client = PathRankingClient()

        if False: #client.is_up() and False:
            triples = [
                ('Mark', 'friendOf', 'John'),
                ('John', 'friendOf', 'Paul'),
                ('Mark', 'friendOfAFriend', 'Paul')]

            pfw_triples = client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)

            for p, f, w in pfw_triples:
                if p == 'friendOfAFriend':
                    self.assertTrue(len(f.hops) == 2)
                    hops = f.hops
                    for hop in hops:
                        self.assertTrue(hop.predicate == 'friendOf' and hop.is_inverse is False)
                    self.assertTrue(w > .0)

    def test_friendship_file(self):
        client = PathRankingClient(url_or_path='data/friendship/rules/friendship.json')

        if False: #client.is_up():
            pfw_triples = client.request(None, threshold=.0, top_k=50)

            for p, f, w in pfw_triples:
                logging.debug("%s :- %s %s" % (p, str(f), w))

    def test_fb15k(self):
        client = PathRankingClient()

        if False: #client.is_up() and False:
            with open('data/fb15k/freebase_mtr100_mte100-train.txt', 'r') as f:
                triples = [tuple(line.split()) for line in f]

            pfw_triples = client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)

            for p, f, w in pfw_triples:
                logging.debug("%s :- %s %s" % (p, str(f), w))

    def test_fb15k_file(self):
        client = PathRankingClient(url_or_path='data/fb15k/rules/fb15k.json.gz')

        if False: #client.is_up() and False:
            pfw_triples = client.request(None, threshold=.0, top_k=10)

            for p, f, w in pfw_triples:
                logging.debug("%s :- %s %s" % (p, str(f), w))

            self.assertTrue(pfw_triples[0][0] == '/film/film/starring./film/performance/actor')
            self.assertTrue(len(pfw_triples[0][1].hops) == 1)
            self.assertAlmostEqual(pfw_triples[0][2], 31.187141996359845)

            logging.debug('Rules retrieved in total: %d' % len(pfw_triples))

    def test_wn18(self):
        client = PathRankingClient()

        if False: #client.is_up() and False:
            with open('data/wn18/wordnet-mlj12-train.txt', 'r') as f:
                triples = [tuple(line.split()) for line in f]

            pfw_triples = client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)

            for p, f, w in pfw_triples:
                logging.debug("%s :- %s %s" % (p, str(f), w))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
