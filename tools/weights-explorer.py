#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pickle

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Model Weights Explorer', formatter_class=formatter)

    argparser.add_argument('weights', action='store', type=str, default=None)
    argparser.add_argument('parser', action='store', type=str, default=None)
    argparser.add_argument('--predicates', '-p', action='store', nargs='+', default=[])

    args = argparser.parse_args(argv)

    weights_path = args.weights
    parser_path = args.parser
    predicates = args.predicates

    with h5py.File(weights_path) as f:
        W = f['/embedding_1/embedding_1_W'][()]
        E = f['/embedding_2/embedding_2_W'][()]

    with open(parser_path, 'rb') as f:
        parser = pickle.load(f)

    predicate_index = parser.predicate_index
    entity_index = parser.entity_index

    idx_lst = [predicate_index[p] for p in predicates]

    for i in range(W.shape[1]):
        print('[%3i]\t%s' % (i, '\t'.join(["%.3f" % W[p_idx][i] for p_idx in idx_lst])))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
