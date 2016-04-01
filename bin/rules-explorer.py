#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hyper.pathranking.api import PathRankingClient

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Rules Explorer', formatter_class=formatter)

    # Rules-related arguments
    argparser.add_argument('--rules', action='store', type=str, default=None,
                           help='JSON document containing the rules extracted from the KG')
    argparser.add_argument('--rules-top-k', action='store', type=int, default=None,
                           help='Top-k rules to consider during the training process')
    argparser.add_argument('--rules-max-length', action='store', type=int, default=None,
                           help='Maximum (body) length for the considered rules')

    args = argparser.parse_args(argv)

    # Rules-related parameters
    rules = args.rules
    rules_top_k = args.rules_top_k
    rules_max_length = args.rules_max_length

    if rules is not None:
        path_ranking_client = PathRankingClient(url_or_path=rules)
        pfw_triples = path_ranking_client.request(None, threshold=.0, top_k=rules_top_k)

        for i, (rule_predicate, rule_feature, rule_weight) in enumerate(pfw_triples):
            if rules_max_length is None or len(rule_feature.hops) <= rules_max_length:
                print('[%d] %s :- %s, %s' % (i, rule_predicate, str(rule_feature), rule_weight))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
