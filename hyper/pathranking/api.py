# -*- coding: utf-8 -*-

import json
import requests

from urllib.parse import urljoin

import logging


class Hop(object):
    def __init__(self, predicate, is_inverse=False):
        self.predicate = predicate
        self.is_inverse = is_inverse

    def __str__(self):
        return self.predicate + ('^-1' if self.is_inverse is True else '')


class Feature(object):
    def __init__(self, hops):
        self.hops = hops

    def __str__(self):
        return ' . '.join([str(hop) for hop in self.hops])


class PredicateFeatures(object):
    def __init__(self, predicate, features, weights):
        self.predicate = predicate
        self.features = features
        self.weights = weights

    def __str__(self):
        sorted_features_weights = reversed(sorted(zip(self.features, self.weights), key=lambda fw: fw[1]))
        return '\n'.join([self.predicate + ' : ' + str(feature) + ' (' + str(weight) + ')'
                          for feature, weight in sorted_features_weights])


class PathRankingClient(object):
    DEFAULT_PRA_PARAMETERS = {
        'features': {
            'type': 'pra',
            'path finder': {
                'type': 'RandomWalkPathFinder',
                'walks per source': 100,
                'path finding iterations': 3,
                'path accept policy': 'paired-only'
            },
            'path selector': {
                'number of paths to keep': 1000
            },
            'path follower': {
                'walks per path': 50,
                'matrix accept policy': 'all-targets'
            }
        },
        'learning': {
            'l1 weight': 0.005,
            'l2 weight': 1
        }
    }

    def __init__(self, url='http://127.0.0.1:8091/'):
        self.service_url = url

    def is_online(self):
        is_online = False
        try:
            ans = requests.get(urljoin(self.service_url, '/status'))
            status = ans.json()
            if status['status'] == 'up':
                is_online = True
        except requests.exceptions.ConnectionError:
            pass
        return is_online

    def request(self, triples, parameters=None, predicates=None):
        request = dict()

        if parameters is not None:
            request['parameters'] = parameters
        if predicates is not None:
            request['predicates'] = predicates
        if triples is not None:
            request['triples'] = triples

        ans = requests.post(urljoin(self.service_url, '/pathRanking'), json=request)

        ans_predicates = []
        for predicate_obj in json.loads(ans.json()):
            ans_predicate = predicate_obj['predicate']

            features, weights = [], []
            for feature_obj in predicate_obj['features']:

                hops = []
                for hop_obj in feature_obj['feature']['hops']:
                    _predicate = hop_obj['predicate']
                    reverse = hop_obj['reverse']
                    hops += [Hop(_predicate, reverse)]

                features += [Feature(hops)]
                weights += [feature_obj['weight']]

            ans_predicates += [PredicateFeatures(ans_predicate, features, weights)]

        return ans_predicates


import sys

if __name__ == '__main__':
    triples = [
        ('Mark', 'friendOf', 'John'),
        ('John', 'friendOf', 'Paul'),
        ('Mark', 'friendOfAFriend', 'Paul')
    ]

    client = PathRankingClient()
    print(client.is_online())

    sys.exit(0)

    predicates = client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)
    for p in predicates:
        print(str(p))
