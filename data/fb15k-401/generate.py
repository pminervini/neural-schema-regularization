#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip

import sys
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


def main(argv):
    train_triples = read_triples('../fb15k/freebase_mtr100_mte100-train.txt')
    valid_triples = read_triples('../fb15k/freebase_mtr100_mte100-valid.txt')
    test_triples = read_triples('../fb15k/freebase_mtr100_mte100-test.txt')

    predicate_counts = {}
    for (s, p, o) in train_triples:
        if p not in predicate_counts:
            predicate_counts[p] = 0
        predicate_counts[p] += 1

    with open('fb15k-401-train.txt', 'w') as f:
        f.writelines(['\t'.join([s, p, o]) + '\n' for (s, p, o) in train_triples if predicate_counts[p] >= 100])

    with open('fb15k-401-valid.txt', 'w') as f:
        f.writelines(['\t'.join([s, p, o]) + '\n' for (s, p, o) in valid_triples if predicate_counts[p] >= 100])

    with open('fb15k-401-test.txt', 'w') as f:
        f.writelines(['\t'.join([s, p, o]) + '\n' for (s, p, o) in test_triples if predicate_counts[p] >= 100])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
