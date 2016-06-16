#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import sys

from collections import Counter
from ascii_graph import Pyasciigraph

import argparse
import logging

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def read_triples(path):
    triples = None
    if path is not None:
        logging.info('Acquiring %s ..' % path)
        my_open = gzip.open if path.endswith('.gz') else open
        with my_open(path, 'rt') as f:
            lines = f.readlines()
        triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split() for l in lines]]
    return triples


def summary(triples):
    entities = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]
    predicates = [p for (_, p, _) in triples]

    entity_counts = sorted(Counter(entities).items(), key=lambda item: item[1])
    graph = Pyasciigraph()
    for line in graph.graph('Entity Counts', entity_counts):
        print(line)


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Show stats on a Knowledge Graphs (entity and predicate frequencies)', formatter_class=formatter)
    argparser.add_argument('triples', nargs='+', help='TSV Files containing triples')
    args = argparser.parse_args(argv)

    triple_paths = args.triples

    triples = sorted([triple for path in triple_paths for triple in read_triples(path)])

    summary(triples)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
