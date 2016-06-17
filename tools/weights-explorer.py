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

    argparser.add_argument('-w', '--weights', required=True, action='store', type=str, default=None)
    argparser.add_argument('-p', '--parser', required=True, action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    weights_path = args.weights
    parser_path = args.parser

    with h5py.File(weights_path) as f:
        W = f['/embedding_1/embedding_1_W'][()]
        E = f['/embedding_2/embedding_2_W'][()]

    with open(parser_path, 'rb') as f:
        parser = pickle.load(f)

    predicate_index = parser.predicate_index
    entity_index = parser.entity_index

    has_part_idx, part_of_idx = predicate_index['_has_part'], predicate_index['_part_of']
    _12493208_idx, _12493426_idx = entity_index['12493208'], entity_index['12493426']

    for i in range(W.shape[1]):
        print('[%3i]\t%.3f\t%.3f\t%.3f\t%.3f'
              % (i, W[has_part_idx][i], W[part_of_idx][i], E[_12493208_idx][i], E[_12493426_idx][i]))

    # 12493208 _has_part 12493426
    # 12493426 _part_of 12493208

    ok, no_ok = 0, 0

    for i in range(W.shape[1]):
        true_dist = abs(E[_12493426_idx][i] + W[part_of_idx][i] - E[_12493208_idx][i])
        false_dist = abs(E[_12493208_idx][i] + W[part_of_idx][i] - E[_12493426_idx][i])

        if true_dist < false_dist:
            ok += 1
        else:
            no_ok += 1

        print('%.2f < %.2f' % (true_dist, false_dist))

    print('ok: %s, no ok: %s' % (ok, no_ok))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
