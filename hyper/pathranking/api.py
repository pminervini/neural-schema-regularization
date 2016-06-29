# -*- coding: utf-8 -*-

import json
import requests

from hyper.pathranking.domain import Hop, Feature
from hyper.io import iopen

from urllib.parse import urljoin
import os.path


class PathRankingClient(object):
    DEFAULT_PRA_PARAMETERS = {
        'features': {
            'type': 'pra',
            'path finder': {
                'type': 'RandomWalkPathFinder',
                'walks per source': 100,
                'path findsing iterations': 3,
                'path accept policy': 'paired-only'
            },
            'path selector': {
                'number of paths to keep': 1000
            },
            'path follower': {
                'walks per path': 50,
                'matrix accept policy': 'paired-targets-only'
            }
        },
        'learning': {
            'l1 weight': 0.005,
            'l2 weight': 1
        }
    }

    def __init__(self, url_or_path='http://127.0.0.1:8091/'):
        self.url_or_path = url_or_path

    def is_up(self):
        is_up = False
        if self.url_or_path.startswith('http'):
            try:
                ans = requests.get(urljoin(self.url_or_path, '/status'))
                status = ans.json()
                if status['status'] == 'up':
                    is_up = True
            except requests.exceptions.ConnectionError:
                pass
        else:
            is_up = os.path.isfile(self.url_or_path)
        return is_up

    def request(self, triples, parameters=None, predicates=None, *args, **kwargs):
        if self.url_or_path.startswith('http'):
            ans = self._http_request(triples, parameters, predicates)
        else:
            with iopen(self.url_or_path, 'r') as f:
                ans = f.read().decode("utf-8") if self.url_or_path.endswith('.gz') else f.read()
        return self._to_pfw_triples(json_str=ans, *args, **kwargs)

    @staticmethod
    def _to_pfw_triples(json_str, threshold=None, top_k=None):
        pfw_triples = []
        for predicate_obj in json.loads(json_str):
            predicate_name = predicate_obj['predicate']

            for feature_obj in predicate_obj['features']:
                weight = feature_obj['weight']
                if threshold is None or weight > threshold:

                    hops = []
                    for hop_obj in feature_obj['feature']['hops']:
                        _predicate = hop_obj['predicate']
                        reverse = hop_obj['reverse']
                        hops += [Hop(_predicate, reverse)]

                    pfw_triples += [(predicate_name, Feature(hops), weight)]

        if top_k is not None:
            sorted_pfw_triples = list(reversed(sorted(pfw_triples, key=lambda x: x[2])))
            pfw_triples = sorted_pfw_triples[:top_k]

        return pfw_triples

    def _http_request(self, triples, parameters=None, predicates=None):
        request = dict()
        if parameters is not None:
            request['parameters'] = parameters
        if predicates is not None:
            request['predicates'] = predicates
        if triples is not None:
            request['triples'] = triples
        ans = requests.post(urljoin(self.url_or_path, '/pathRanking'), json=request)
        return ans.json()


if __name__ == '__main__':
    triples = [
        ('Mark', 'friendOf', 'John'),
        ('John', 'friendOf', 'Paul'),
        ('Mark', 'friendOfAFriend', 'Paul')
    ]

    client = PathRankingClient()
    if client.is_up():
        predicates = client.request(triples, parameters=PathRankingClient.DEFAULT_PRA_PARAMETERS)
        for p in predicates:
            print(str(p))
