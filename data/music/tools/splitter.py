#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from tqdm import tqdm

import argparse
import logging
import sys

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Knowledge Graph Splitter', formatter_class=formatter)

    argparser.add_argument('--kb', required=True, type=argparse.FileType('r'), default='/dev/stdin',
                           help='Path of the KB')

    argparser.add_argument('--train', required=True, type=argparse.FileType('w'), default='/dev/stdout',
                           help='Path of the training set')

    argparser.add_argument('--validation', required=True, type=argparse.FileType('w'), default='/dev/stdout',
                           help='Path of the validation set')
    argparser.add_argument('--validation-size', required=True, action='store', type=int, default=10000,
                           help='Size of the validation set')

    argparser.add_argument('--test', required=True, type=argparse.FileType('w'), default='/dev/stdout',
                           help='Path of the test set')
    argparser.add_argument('--test-size', required=True, action='store', type=int, default=10000,
                           help='Size of the test set')

    argparser.add_argument('--seed', action='store', type=int, default=0,
                           help='Seed for the PRNG')

    args = argparser.parse_args(argv)

    kb_fd = args.kb

    train_fd = args.train

    valid_fd = args.validation
    valid_size = args.validation_size
    assert valid_size > 0

    test_fd = args.test
    test_size = args.test_size
    assert test_size > 0

    seed = args.seed

    logging.debug('Importing the Knowledge Graph ..')

    _triples, _symbols = [], set()
    for line in tqdm(kb_fd):
        s, p, o, nl = line.split(' ')
        _triples += [(s, p, o)]
        _symbols |= {s, p, o}

    _sym2idx = {sym: idx for idx, sym in enumerate(_symbols)}
    _idx2sym = {idx: sym for sym, idx in _sym2idx.items()}

    _idx_triples = []
    for _triple in _triples:
        s, p, o = _triple
        _idx_triples += [(_sym2idx[s], _sym2idx[p], _sym2idx[o])]

    kb_triples = np.array(_idx_triples)

    NT = kb_triples.shape[0]

    logging.debug('Number of triples in the Knowledge Graph: %s' % NT)

    train_size = NT - (valid_size + test_size)
    assert train_size > 0

    logging.debug('Generating a random permutation of RDF triples ..')

    random_state = np.random.RandomState(seed=seed)
    permutation = random_state.permutation(NT)

    shuffled_triples = kb_triples[permutation]

    logging.debug('Building the training, validation and test sets ..')

    train_triples = shuffled_triples[:train_size]
    valid_triples = shuffled_triples[train_size:][:valid_size]
    test_triples = shuffled_triples[train_size:][valid_size:][:test_size]

    logging.debug('Saving ..')

    train_fd.writelines(['\t'.join(_idx2sym[idx] for idx in triple) + '\n' for triple in train_triples])
    valid_fd.writelines(['\t'.join(_idx2sym[idx] for idx in triple) + '\n' for triple in valid_triples])
    test_fd.writelines(['\t'.join(_idx2sym[idx] for idx in triple) + '\n' for triple in test_triples])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
