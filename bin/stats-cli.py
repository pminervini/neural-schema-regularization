#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from ascii_graph import Pyasciigraph

from hyper.io import read_triples

import sys
import argparse
import logging

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def summary(triples):
    entities = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]
    predicates = [p for (_, p, _) in triples]

    for sequence in [entities, predicates]:
        item_counts = sorted(Counter(sequence).items(), key=lambda item: item[1])
        graph = Pyasciigraph()
        for line in graph.graph('Counts', item_counts):
            print(line)


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Show stats on a Knowledge Graphs', formatter_class=formatter)
    argparser.add_argument('triples', nargs='+', help='TSV Files containing triples')
    args = argparser.parse_args(argv)

    triple_paths = args.triples

    triples = sorted([triple for path in triple_paths for triple in read_triples(path)])

    summary(triples)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
